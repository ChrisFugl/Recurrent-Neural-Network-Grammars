from app.tasks.task import Task
import os
import time
from torch.utils.tensorboard import SummaryWriter

class TrainTask(Task):

    def __init__(self, device, iterator_train, iterator_val, model, loss, optimizer, stopping_criterion):
        """
        :type device: torch.device
        :type iterator_train: app.data.iterators.iterator.Iterator
        :type iterator_val: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type loss: app.losses.loss.Loss
        :type optimizer: torch.optim.Optimizer
        :type stopping_criterion: app.stopping_criteria.stopping_criterion.StoppingCriterion
        """
        super().__init__()
        self._device = device
        self._iterator_train = iterator_train
        self._iterator_val = iterator_val
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._stopping_criterion = stopping_criterion

        working_directory = os.getcwd()
        tensorboard_directory = os.path.join(working_directory, 'tb')
        train_directory = os.path.join(tensorboard_directory, 'train')
        val_directory = os.path.join(tensorboard_directory, 'val')
        self._writer_train = SummaryWriter(log_dir=train_directory)
        self._writer_val = SummaryWriter(log_dir=val_directory)

    def run(self):
        time_start = time.time()
        batch_count = 0
        epoch = 0
        loss_val = self._validate(batch_count)
        print(f'Epoch=0, loss_val={loss_val:0.4f}')
        self._writer_train.add_scalar('epoch', epoch, batch_count)
        while not self._stopping_criterion.is_done(epoch, loss_val):
            epoch += 1
            batch_count, loss_train = self._train_epoch(time_start, epoch, batch_count)
            loss_val = self._validate(batch_count)
            print(f'Epoch={epoch}, loss_train={loss_train:0.4f} loss_val={loss_val:0.4f}')

    def _optimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _train_epoch(self, time_start, epoch, batch_count):
        time_epoch_start = time.time()
        for batch in self._iterator_train:
            batch_count += len(batch[1][1])
            loss_train = self._train_batch(batch, batch_count)
        time_epoch_stop = time.time()
        self._writer_train.add_scalar('epoch', epoch, batch_count)
        self._writer_train.add_scalar('time_s/epoch', self._get_seconds(time_epoch_start, time_epoch_stop), batch_count)
        self._writer_train.add_scalar('time_s/total', self._get_seconds(time_start, time_epoch_stop), batch_count)
        return batch_count, loss_train

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
