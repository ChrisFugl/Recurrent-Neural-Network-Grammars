from app.tasks.task import Task
import torch

class TrainTask(Task):

    def __init__(self,
        device,
        iterator_train,
        iterator_val,
        model,
        loss,
        optimizer,
        token_converter,
        action_converter,
        stopping_criterion
    ):
        """
        :type device: torch.device
        :type iterator_train: app.data.iterators.iterator.Iterator
        :type iterator_val: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type loss: app.losses.loss.Loss
        :type optimizer: torch.optim.Optimizer
        :type token_converter: app.data.converters.token.TokenConverter
        :type action_converter: app.data.converters.action.ActionConverter
        :type stopping_criterion: app.stopping_criteria.stopping_criterion.StoppingCriterion
        """
        super().__init__()
        self._device = device
        self._iterator_train = iterator_train
        self._iterator_val = iterator_val
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._token_converter = token_converter
        self._action_converter = action_converter
        self._stopping_criterion = stopping_criterion

    def run(self):
        epoch = 1
        loss_val = self._validate()
        print(f'Epoch=0, loss_val={loss_val:0.4f}')
        while not self._stopping_criterion.is_done(epoch, loss_val):
            for batch_tokens, batch_actions in self._iterator_train:
                _, batch_actions_lengths, _ = batch_actions
                batch_probabilities = self._model(batch_tokens, batch_actions)
                loss_batch = self._loss(batch_probabilities, batch_actions_lengths)
                self._optimizer.zero_grad()
                loss_batch.backward()
                self._optimizer.step()
            loss_val = self._validate()
            print(f'Epoch={epoch}, loss_val={loss_val:0.4f}')
            epoch += 1

    def _validate(self):
        self._model.eval()
        losses = []
        for tokens, actions in self._iterator_val:
            _, actions_lengths, _ = actions
            actions_probabilities = self._model(tokens, actions)
            loss = self._loss(actions_probabilities, actions_lengths)
            losses.append(loss)
        loss_val = sum(losses) / len(losses)
        self._model.train()
        return loss_val
