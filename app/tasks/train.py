from app.tasks.task import Task
import hydra
import logging
import numpy as np
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
        log_train_every, load_checkpoint, token_count, tag_count, non_terminal_count, action_count
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
        :type log_train_every: int
        :type load_checkpoint: str
        :type token_count: int
        :type tag_count: int
        :type non_terminal_count: int
        :type action_count: int
        """
        super().__init__()
        self.device = device
        self.iterator_train = iterator_train
        self.iterator_val = iterator_val
        self.model = model
        self.loss = loss
        self.learning_rate_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.stopping_criterion = stopping_criterion
        self.checkpoint = checkpoint
        self.evaluator = evaluator
        self.log_train_every = log_train_every

        self.token_count = token_count
        self.tag_count = tag_count
        self.non_terminal_count = non_terminal_count
        self.action_count = action_count

        working_directory = os.getcwd()
        tensorboard_directory = os.path.join(working_directory, 'tb')
        train_directory = os.path.join(tensorboard_directory, 'train')
        val_directory = os.path.join(tensorboard_directory, 'val')
        self.writer_train = SummaryWriter(log_dir=train_directory)
        self.writer_val = SummaryWriter(log_dir=val_directory)

        self.total_time_offset = 0
        self.start_epoch = 0
        self.start_batch_count = 0

        self.logger = logging.getLogger('train')

        self.uses_gpu = self.device.type == 'cuda'

        if load_checkpoint is not None:
            self.load_checkpoint(load_checkpoint)

    def run(self):
        time_start = time.time()
        self.logger.info('Starting training')
        self.logger.info(f'Saving output to {os.getcwd()}')
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Tokens: {self.token_count:,}')
        self.logger.info(f'Part-of-speech tags: {self.tag_count:,}')
        self.logger.info(f'Non-terminals: {self.non_terminal_count:,}')
        self.logger.info(f'Actions: {self.action_count:,}')
        self.logger.info(f'Parameters: {self.count_parameters():,}')
        self.logger.info(f'Model:\n{self.model}')
        batch_count = self.start_batch_count
        epoch = self.start_epoch
        losses = []
        best_loss_val = None
        self.model.train()
        if self.evaluator.should_evaluate(epoch, batch_count, pretraining=True):
            best_loss_val = self.evaluate(epoch, batch_count, best_loss_val)
        # assumes a single learning rate for all parameters
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.log_epoch(epoch, batch_count, learning_rate)
        while True:
            epoch, batch_count, done, losses, best_loss_val = self.train_epoch(time_start, epoch, batch_count, losses, best_loss_val)
            if done:
                break
            if self.evaluator.should_evaluate(epoch, batch_count, end_of_epoch=True):
                best_loss_val = self.evaluate(epoch, batch_count, best_loss_val)
            self.stopping_criterion.add_epoch(epoch)
            if self.stopping_criterion.is_done():
                break
        self.evaluate(epoch, batch_count, best_loss_val)
        time_stop = time.time()
        time_seconds = self.get_seconds(time_start, time_stop)
        time_hours = time_seconds / 3600
        time_days = time_hours / 24
        self.logger.info(f'Training time: {time_seconds:0.2f} s/{time_hours:0.2f} h/{time_days:0.2f} d')
        self.logger.info('Finished training')

    def train_epoch(self, time_start, epoch, batch_count, losses, best_loss_val):
        time_epoch_start = time.time()
        start_of_epoch = True
        for batch in self.iterator_train:
            batch_count += 1
            loss_train = self.train_batch(batch, batch_count)
            losses.append(loss_train)
            if batch_count % self.log_train_every == 0:
                self.log_loss(self.writer_train, batch_count, losses)
                losses = []
            if self.evaluator.should_evaluate(epoch, batch_count):
                best_loss_val = self.evaluate(epoch, batch_count, best_loss_val)
            if self.stopping_criterion.is_done():
                return epoch, batch_count, True, losses, best_loss_val
            if self.checkpoint.should_save_checkpoint(epoch, batch_count, start_of_epoch):
                self.save_checkpoint(time_start, epoch, batch_count)
            start_of_epoch = False
        self.learning_rate_scheduler.step()
        # assumes a single learning rate for all parameters
        learning_rate = self.learning_rate_scheduler.get_last_lr()[0]
        epoch += 1
        time_epoch_stop = time.time()
        time_total = self.total_time_offset + self.get_seconds(time_start, time_epoch_stop)
        self.log_epoch(epoch, batch_count, learning_rate)
        self.writer_train.add_scalar('time/epoch_s', self.get_seconds(time_epoch_start, time_epoch_stop), batch_count)
        self.writer_train.add_scalar('time/total_s', time_total, batch_count)
        return epoch, batch_count, False, losses, best_loss_val

    def train_batch(self, batch, batch_count):
        self.start_measure_memory()
        time_batch_start = time.time()
        batch_log_probs = self.model.batch_log_likelihood(batch)
        loss = self.loss(batch_log_probs, batch.actions.tensor, batch.actions.lengths)
        time_batch_stop = time.time()
        time_optimize_start = time.time()
        self.optimize(loss)
        loss_scalar = loss.cpu().item()
        time_optimize_stop = time.time()
        time_batch = self.get_seconds(time_batch_start, time_batch_stop)
        time_optimize = self.get_seconds(time_optimize_start, time_optimize_stop)
        actions_per_second = self.count_actions(batch) / time_batch
        tokens_per_second = self.count_tokens(batch) / time_batch
        sentences_per_second = batch.size / time_batch
        self.writer_train.add_scalar('time/actions_per_s', actions_per_second, batch_count)
        self.writer_train.add_scalar('time/tokens_per_s', tokens_per_second, batch_count)
        self.writer_train.add_scalar('time/sentences_per_s', sentences_per_second, batch_count)
        self.writer_train.add_scalar('time/batch_s', time_batch, batch_count)
        self.writer_train.add_scalar('time/optimize_s', time_optimize, batch_count)
        self.writer_train.add_scalar('batch_stats/action_count_min', batch.actions.lengths.min(), batch_count)
        self.writer_train.add_scalar('batch_stats/action_count_mean', sum(batch.actions.lengths) / len(batch.actions.lengths), batch_count)
        self.writer_train.add_scalar('batch_stats/action_count_max', batch.max_actions_length, batch_count)
        self.writer_train.add_scalar('batch_stats/token_count_min', batch.tokens.lengths.min(), batch_count)
        self.writer_train.add_scalar('batch_stats/token_count_mean', sum(batch.tokens.lengths) / len(batch.tokens.lengths), batch_count)
        self.writer_train.add_scalar('batch_stats/token_count_max', batch.max_tokens_length, batch_count)
        self.log_memory(self.writer_train, batch_count)
        return loss_scalar

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, epoch, batch_count, best_loss_val):
        self.start_measure_memory()
        time_val_start = time.time()
        self.model.eval()
        losses = []
        total_actions = 0
        total_tokens = 0
        total_sentences = self.iterator_val.size()
        for batch in self.iterator_val:
            log_probs = self.model.batch_log_likelihood(batch)
            loss = self.loss(log_probs, batch.actions.tensor, batch.actions.lengths)
            loss_scalar = loss.cpu().item()
            losses.append(loss_scalar)
            total_actions += self.count_actions(batch)
            total_tokens += self.count_tokens(batch)
        self.model.train()
        time_val_stop = time.time()
        self.log_memory(self.writer_val, batch_count)
        time_val = self.get_seconds(time_val_start, time_val_stop)
        loss_val = self.log_loss(self.writer_val, batch_count, losses)
        self.stopping_criterion.add_val_loss(loss_val)
        self.logger.info(f'epoch={epoch}, batch={batch_count}, loss_val={loss_val:0.8f}')
        actions_per_second = total_actions / time_val
        tokens_per_second = total_tokens / time_val
        sentences_per_second = total_sentences / time_val
        self.writer_val.add_scalar('time/val_s', time_val, batch_count)
        self.writer_val.add_scalar('time/actions_per_s', actions_per_second, batch_count)
        self.writer_val.add_scalar('time/tokens_per_s', tokens_per_second, batch_count)
        self.writer_val.add_scalar('time/sentences_per_s', sentences_per_second, batch_count)

        # save model with best performing validation loss
        if best_loss_val is None or loss_val < best_loss_val:
            if best_loss_val is not None:
                self.save()
            return loss_val
        else:
            return best_loss_val

    def get_seconds(self, start, stop):
        seconds = stop - start
        return seconds

    def save(self):
        file_name = 'model.pt'
        directory = hydra.utils.to_absolute_path(os.getcwd())
        file_path = os.path.join(directory, file_name)
        self.logger.info(f'Saving model at {file_path}')
        self.model.save(file_path)

    def save_checkpoint(self, time_start, epoch, batch_count):
        time_checkpoint_start = time.time()
        checkpoints_directory = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(checkpoints_directory, exist_ok=True)
        checkpoint_name = f'epoch_{epoch}_batch_{batch_count}.pt'
        checkpoint_path = os.path.join(checkpoints_directory, checkpoint_name)
        self.logger.info(f'Saving checkpoint at {checkpoint_path}')
        checkpoint = {
            'batch_count': batch_count,
            'epoch': epoch,
            'learning_rate_scheduler': self.learning_rate_scheduler.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stopping_criterion': self.stopping_criterion.state_dict(),
            'time_elapsed': self.get_seconds(time_start, time.time()),
        }
        torch.save(checkpoint, checkpoint_path)
        time_checkpoint_stop = time.time()
        self.writer_train.add_scalar('time/checkpoint_s', self.get_seconds(time_checkpoint_start, time_checkpoint_stop), batch_count)

    def load_checkpoint(self, path):
        absolute_path = hydra.utils.to_absolute_path(path)
        self.logger.info(f'Loading checkpoint from {absolute_path}')
        checkpoint = torch.load(absolute_path)
        self.start_batch_count = checkpoint['batch_count']
        self.start_epoch = checkpoint['epoch']
        self.total_time_offset = checkpoint['time_elapsed']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['learning_rate_scheduler'])
        self.stopping_criterion.load_state_dict(checkpoint['stopping_criterion'])

    def log_epoch(self, epoch, batch_count, learning_rate):
        self.logger.info(f'epoch={epoch}, batch={batch_count}, learning rate={learning_rate}')
        self.writer_train.add_scalar('training/epoch', epoch, batch_count)
        self.writer_train.add_scalar('training/learning_rate', learning_rate, batch_count)

    def count_parameters(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        count = sum([np.prod(parameter.size()) for parameter in parameters])
        return count

    def byte2gb(self, byte_amount):
        return byte_amount / (1024 ** 3)

    def start_measure_memory(self):
        if self.uses_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

    def log_memory(self, writer, batch_count):
        if self.uses_gpu:
            allocated_gb = self.byte2gb(torch.cuda.max_memory_allocated(self.device))
            reserved_gb = self.byte2gb(torch.cuda.max_memory_reserved(self.device))
            writer.add_scalar('memory/allocated_gb', allocated_gb, batch_count)
            writer.add_scalar('memory/reserved_gb', reserved_gb, batch_count)

    def count_actions(self, batch):
        return sum(map(len, batch.actions.actions))

    def count_tokens(self, batch):
        return sum(map(len, batch.tokens.tokens))

    def log_loss(self, writer, batch_count, losses):
        loss = sum(losses) / len(losses)
        writer.add_scalar('training/loss', loss, batch_count)
        return loss
