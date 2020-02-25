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
        model, loss, optimizer,
        stopping_criterion, checkpoint,
        load_checkpoint,
    ):
        """
        :type device: torch.device
        :type iterator_train: app.data.iterators.iterator.Iterator
        :type iterator_val: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type loss: app.losses.loss.Loss
        :type optimizer: torch.optim.Optimizer
        :type stopping_criterion: app.stopping_criteria.stopping_criterion.StoppingCriterion
        :type checkpoint: app.checkpoints.checkpoint.Checkpoint
        :type load_checkpoint: str
        """
        super().__init__()
        self._device = device
        self._iterator_train = iterator_train
        self._iterator_val = iterator_val
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._stopping_criterion = stopping_criterion
        self._checkpoint = checkpoint

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
        self._logger.info(f'Saving output in {os.getcwd()}')
        batch_count = self._start_batch_count
        epoch = self._start_epoch
        loss_val = self._validate(batch_count)
        self._logger.info(f'Epoch={epoch}, loss_val={loss_val:0.4f}')
        self._writer_train.add_scalar('epoch', epoch, batch_count)
        while not self._stopping_criterion.is_done(epoch, loss_val):
            epoch, batch_count, loss_train = self._train_epoch(time_start, epoch, batch_count)
            loss_val = self._validate(batch_count)
            self._logger.info(f'Epoch={epoch}, loss_train={loss_train:0.4f} loss_val={loss_val:0.4f}')
        self._save()

    def _train_epoch(self, time_start, epoch, batch_count):
        time_epoch_start = time.time()
        start_of_epoch = True
        for batch in self._iterator_train:
            batch_count += 1
            loss_train = self._train_batch(batch, batch_count)
            if self._checkpoint.should_save_checkpoint(epoch, batch_count, start_of_epoch):
                self._save_checkpoint(time_start, epoch, batch_count)
            start_of_epoch = False
        epoch += 1
        time_epoch_stop = time.time()
        time_total = self._total_time_offset + self._get_seconds(time_start, time_epoch_stop)
        self._writer_train.add_scalar('epoch', epoch, batch_count)
        self._writer_train.add_scalar('time_s/epoch', self._get_seconds(time_epoch_start, time_epoch_stop), batch_count)
        self._writer_train.add_scalar('time_s/total', time_total, batch_count)
        return epoch, batch_count, loss_train

    def _train_batch(self, batch, batch_count):
        time_batch_start = time.time()
        batch_tokens, batch_actions = batch
        _, batch_actions_lengths, _ = batch_actions
        batch_log_probs = self._model(batch_tokens, batch_actions)
        loss_train = self._loss(batch_log_probs, batch_actions_lengths)
        self._optimize(loss_train)
        time_batch_stop = time.time()
        self._writer_train.add_scalar('loss', loss_train, batch_count)
        self._writer_train.add_scalar('time_s/batch', self._get_seconds(time_batch_start, time_batch_stop), batch_count)
        return loss_train

    def _optimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _validate(self, batch_count):
        time_val_start = time.time()
        self._model.eval()
        losses = []
        for tokens, actions in self._iterator_val:
            _, actions_lengths, _ = actions
            log_probs = self._model(tokens, actions)
            loss = self._loss(log_probs, actions_lengths)
            losses.append(loss)
        loss_val = sum(losses) / len(losses)
        time_val_stop = time.time()
        self._writer_val.add_scalar('loss', loss_val, batch_count)
        self._writer_val.add_scalar('time_s/val', self._get_seconds(time_val_start, time_val_stop), batch_count)
        self._model.train()
        return loss_val

    def _get_seconds(self, start, stop):
        seconds = stop - start
        return seconds

    def _save(self):
        file_name = 'model.pt'
        directory = hydra.utils.to_absolute_path(os.getcwd())
        file_path = os.path.join(directory, file_name)
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
        self._stopping_criterion.load_state_dict(checkpoint['stopping_criterion'])
