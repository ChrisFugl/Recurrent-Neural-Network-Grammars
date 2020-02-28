from app.tasks.task import Task
import hydra
import logging
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainTask(Task):

    def __init__(self,
        device,
        iterator_train, iterator_val,
        model, loss, optimizer, learning_rate_scheduler,
        stopping_criterion, checkpoint, evaluator,
        load_checkpoint, token_count, non_terminal_count, action_count
    ):
        """
        :type device: torch.device
        :type iterator_train: app.data.iterators.iterator.Iterator
        :type iterator_val: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type loss: app.losses.loss.Loss
        :type learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler
        :type optimizer: torch.optim.Optimizer
        :type stopping_criterion: app.stopping_criteria.stopping_criterion.StoppingCriterion
        :type checkpoint: app.checkpoints.checkpoint.Checkpoint
        :type evaluator: app.evaluators.evaluator.Evaluator
        :type load_checkpoint: str
        :type token_count: int
        :type non_terminal_count: int
        :type action_count: int
        """
        super().__init__()
        self._device = device
        self._iterator_train = iterator_train
        self._iterator_val = iterator_val
        self._model = model
        self._loss = loss
        self._learning_rate_scheduler = learning_rate_scheduler
        self._optimizer = optimizer
        self._stopping_criterion = stopping_criterion
        self._checkpoint = checkpoint
        self._evaluator = evaluator

        self._token_count = token_count
        self._non_terminal_count = non_terminal_count
        self._action_count = action_count

        working_directory = os.getcwd()
        tensorboard_directory = os.path.join(working_directory, 'tb')
        train_directory = os.path.join(tensorboard_directory, 'train')
        val_directory = os.path.join(tensorboard_directory, 'val')
        self._writer_train = SummaryWriter(log_dir=train_directory)
        self._writer_val = SummaryWriter(log_dir=val_directory)

        self._total_time_offset = 0
        self._start_epoch = 0
        self._start_batch_count = 0

        self._logger = logging.getLogger('train')

        if load_checkpoint is not None:
            self._load_checkpoint(load_checkpoint)

    def run(self):
        time_start = time.time()
        self._logger.info('Starting training')
        self._logger.info(f'Saving output to {os.getcwd()}')
        self._logger.info(f'Using device: {self._device}')
        self._logger.info(f'Tokens: {self._token_count}')
        self._logger.info(f'Non-terminals: {self._non_terminal_count}')
        self._logger.info(f'Actions: {self._action_count}')
        batch_count = self._start_batch_count
        epoch = self._start_epoch
        if self._evaluator.should_evaluate(epoch, batch_count, pretraining=True):
            self._evaluate(epoch, batch_count)
        # assumes a single learning rate for all parameters
        learning_rate = self._optimizer.param_groups[0]['lr']
        self._log_epoch(epoch, batch_count, learning_rate)
        while True:
            epoch, batch_count, done = self._train_epoch(time_start, epoch, batch_count)
            if done:
                break
            if self._evaluator.should_evaluate(epoch, batch_count, end_of_epoch=True):
                self._evaluate(epoch, batch_count)
            self._stopping_criterion.add_epoch(epoch)
            if self._stopping_criterion.is_done():
                break
        if self._evaluator.should_evaluate(epoch, batch_count, posttraining=True):
            self._evaluate(epoch, batch_count)
        self._save()
        time_stop = time.time()
        time_seconds = self._get_seconds(time_start, time_stop)
        time_hours = time_seconds / 3600
        time_days = time_hours / 24
        self._logger.info(f'Training time: {time_seconds:0.2f} s/{time_hours:0.2f} h/{time_days:0.2f} d')
        self._logger.info('Finished training')

    def _train_epoch(self, time_start, epoch, batch_count):
        time_epoch_start = time.time()
        start_of_epoch = True
        for batch in self._iterator_train:
            batch_count += 1
            self._train_batch(batch, batch_count)
            if self._evaluator.should_evaluate(epoch, batch_count):
                self._evaluate(epoch, batch_count)
                # check if stopping criterion necessitates an early stopping
                if self._stopping_criterion.is_done():
                    return epoch, batch_count, True
            if self._checkpoint.should_save_checkpoint(epoch, batch_count, start_of_epoch):
                self._save_checkpoint(time_start, epoch, batch_count)
            start_of_epoch = False
        self._learning_rate_scheduler.step()
        # assumes a single learning rate for all parameters
        learning_rate = self._learning_rate_scheduler.get_last_lr()[0]
        epoch += 1
        time_epoch_stop = time.time()
        time_total = self._total_time_offset + self._get_seconds(time_start, time_epoch_stop)
        self._log_epoch(epoch, batch_count, learning_rate)
        self._writer_train.add_scalar('time_s/epoch', self._get_seconds(time_epoch_start, time_epoch_stop), batch_count)
        self._writer_train.add_scalar('time_s/total', time_total, batch_count)
        return epoch, batch_count, False

    def _train_batch(self, batch, batch_count):
        time_batch_start = time.time()
        batch_tokens, batch_actions = batch
        _, batch_actions_lengths, _ = batch_actions
        batch_log_probs = self._model(batch_tokens, batch_actions)
        loss_train = self._loss(batch_log_probs, batch_actions_lengths)
        self._optimize(loss_train)
        time_batch_stop = time.time()
        self._writer_train.add_scalar('training/loss', loss_train, batch_count)
        self._writer_train.add_scalar('time_s/batch', self._get_seconds(time_batch_start, time_batch_stop), batch_count)
        return loss_train

    def _optimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _evaluate(self, epoch, batch_count):
        time_val_start = time.time()
        self._model.eval()
        losses = []
        for tokens, actions in self._iterator_val:
            _, actions_lengths, _ = actions
            log_probs = self._model(tokens, actions)
            loss = self._loss(log_probs, actions_lengths)
            losses.append(loss)
        loss_val = sum(losses) / len(losses)
        self._model.train()
        time_val_stop = time.time()
        self._writer_val.add_scalar('training/loss', loss_val, batch_count)
        self._writer_val.add_scalar('time_s/val', self._get_seconds(time_val_start, time_val_stop), batch_count)
        self._stopping_criterion.add_val_loss(loss_val)
        self._logger.info(f'epoch={epoch}, batch={batch_count}, loss_val={loss_val:0.8f}')

    def _get_seconds(self, start, stop):
        seconds = stop - start
        return seconds

    def _save(self):
        file_name = 'model.pt'
        directory = hydra.utils.to_absolute_path(os.getcwd())
        file_path = os.path.join(directory, file_name)
        self._logger.info(f'Saving model at {file_path}')
        self._model.save(file_path)

    def _save_checkpoint(self, time_start, epoch, batch_count):
        time_checkpoint_start = time.time()
        checkpoints_directory = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(checkpoints_directory, exist_ok=True)
        checkpoint_name = f'epoch_{epoch}_batch_{batch_count}.pt'
        checkpoint_path = os.path.join(checkpoints_directory, checkpoint_name)
        self._logger.info(f'Saving checkpoint at {checkpoint_path}')
        checkpoint = {
            'batch_count': batch_count,
            'epoch': epoch,
            'learning_rate_scheduler': self._learning_rate_scheduler.state_dict(),
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'stopping_criterion': self._stopping_criterion.state_dict(),
            'time_elapsed': self._get_seconds(time_start, time.time()),
        }
        torch.save(checkpoint, checkpoint_path)
        time_checkpoint_stop = time.time()
        self._writer_train.add_scalar('time_s/checkpoint', self._get_seconds(time_checkpoint_start, time_checkpoint_stop), batch_count)

    def _load_checkpoint(self, path):
        absolute_path = hydra.utils.to_absolute_path(path)
        self._logger.info(f'Loading checkpoint from {absolute_path}')
        checkpoint = torch.load(absolute_path)
        self._start_batch_count = checkpoint['batch_count']
        self._start_epoch = checkpoint['epoch']
        self._total_time_offset = checkpoint['time_elapsed']
        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._learning_rate_scheduler.load_state_dict(checkpoint['learning_rate_scheduler'])
        self._stopping_criterion.load_state_dict(checkpoint['stopping_criterion'])

    def _log_epoch(self, epoch, batch_count, learning_rate):
        self._logger.info(f'epoch={epoch}, batch={batch_count}, learning rate={learning_rate}')
        self._writer_train.add_scalar('training/epoch', epoch, batch_count)
        self._writer_train.add_scalar('training/learning_rate', learning_rate, batch_count)
