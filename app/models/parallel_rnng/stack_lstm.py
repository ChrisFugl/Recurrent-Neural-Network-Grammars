from app.models.parallel_rnng.multi_layer_lstm_cell import MultiLayerLSTMCell
from app.utils import batched_index_select
import torch
from torch import nn

class StackLSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = MultiLayerLSTMCell(input_size, hidden_size, num_layers, bias, dropout)

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: stack contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_index = torch.max(self.pos)
        last_layer_state = self.hidden_stack[:, :, :, self.num_layers - 1]
        # do not include the initial state
        contents = last_layer_state[1:max_index + 1, :, :]
        lengths = self.pos
        return contents, lengths

    def initialize(self, stack_size, batch_size):
        """
        :type stack_size: int
        :type batch_size: int
        """
        self.pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        shape = (stack_size + 1, batch_size, self.hidden_size, self.num_layers)
        self.hidden_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.cell_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.batch_indices = torch.arange(0, batch_size, device=self.device, dtype=torch.long)

    def hold_or_pop(self, op):
        """
        :param op: tensor, (batch size), hold = 0, pop = -1
        :type op: torch.Tensor
        :rtype: torch.Tensor
        """
        output = self.top()
        self.pos = self.pos + op
        return output

    def hold_or_push(self, input, op):
        """
        :param input: tensor, (sequence length, batch size, input size)
        :type input: torch.Tensor
        :param op: tensor, (batch size), push = 1, hold = 0
        :type op: torch.Tensor
        """
        batch_size = input.size(0)
        hidden_state = batched_index_select(self.hidden_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        cell_state = batched_index_select(self.cell_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        next_hidden, next_cell = self.lstm(input, (hidden_state, cell_state))
        self.hidden_stack = self.hidden_stack.clone()
        self.cell_stack = self.cell_stack.clone()
        self.hidden_stack[self.pos + 1, self.batch_indices, :, :] = next_hidden
        self.cell_stack[self.pos + 1, self.batch_indices, :, :] = next_cell
        self.pos = self.pos + op

    def top(self):
        """
        :rtype: torch.Tensor
        """
        batch_size = len(self.pos)
        output = batched_index_select(self.hidden_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        output = output[:, :, self.num_layers - 1]
        return output

    def inference_initialize(self, batch_size):
        """
        :type batch_size: int
        :rtype: app.models.parallel_rnng.stack_lstm.StackState
        """
        state_shape = (batch_size, self.hidden_size, self.num_layers)
        hidden = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        cell = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        previous = None
        lengths = [1] * batch_size
        actions = None
        return StackState(batch_size, hidden, cell, previous, lengths, actions)

    def inference_hold_or_pop(self, state, ops):
        """
        :type state: app.models.parallel_rnng.stack_lstm.StackState
        :type ops: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.StackState, torch.Tensor
        """
        output = self.inference_top(state)
        state_shape = (state.batch_size, self.hidden_size, self.num_layers)
        next_hidden = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        next_cell = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        next_previous = []
        next_lengths = [length if op == 0 else length - 1 for length, op in zip(state.lengths, ops)]
        next_actions = [state.actions[i] if op == 0 else state.previous[i].actions[i] for i, op in enumerate(ops)]
        for batch_index, op in enumerate(ops):
            prev_state = state.previous[batch_index]
            if op == 0:
                next_hidden[batch_index] = state.hidden[batch_index]
                next_cell[batch_index] = state.cell[batch_index]
                next_previous.append(prev_state)
            else:
                next_hidden[batch_index] = prev_state.hidden[batch_index]
                next_cell[batch_index] = prev_state.cell[batch_index]
                next_previous.append(prev_state.previous[batch_index])
        next_state = StackState(state.batch_size, next_hidden, next_cell, next_previous, next_lengths, next_actions)
        return next_state, output

    def inference_hold_or_push(self, state, actions, input, ops):
        """
        :type state: app.models.parallel_rnng.stack_lstm.StackState
        :type actions: list of app.data.actions.action.Action
        :type inputs: torch.Tensor
        :type ops: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.StackState, torch.Tensor
        """
        next_hidden, next_cell = self.lstm(input, (state.hidden, state.cell))
        next_previous = [state.previous[index] if op == 0 else state for index, op in enumerate(ops)]
        next_lengths = [length if op == 0 else length + 1 for length, op in zip(state.lengths, ops)]
        next_actions = [state.actions[index] if op == 0 else actions[index] for index, op in enumerate(ops)]
        next_state = StackState(state.batch_size, next_hidden, next_cell, next_previous, next_lengths, next_actions)
        return next_state

    def inference_top(self, state):
        """
        :type state: app.models.parallel_rnng.stack_lstm.StackState
        :rtype: torch.Tensor
        """
        output = state.hidden[:, :, self.num_layers - 1]
        return output

    def inference_contents(self, state):
        """
        :type state: app.models.parallel_rnng.stack_lstm.StackState
        :returns: stack contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_length = max(state.lengths)
        contents = torch.zeros((max_length, state.batch_size, self.hidden_size), device=self.device, dtype=torch.float)
        for batch_index in range(state.batch_size):
            node = state
            length = state.lengths[batch_index]
            while True:
                output = node.hidden[batch_index, :, self.num_layers - 1]
                contents[length - 1, batch_index] = output
                length -= 1
                if length == 0:
                    assert node.previous is None
                    break
                node = node.previous[batch_index]
        lengths = torch.tensor(state.lengths, device=self.device, dtype=torch.long)
        return contents, lengths

    def __str__(self):
        return f'StackLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'

class StackState:

    def __init__(self, batch_size, hidden, cell, previous, lengths, actions):
        """
        :type batch_size: int
        :type hidden: torch.Tensor
        :type cell: torch.Tensor
        :type previous: list of app.models.parallel_rnng.stack_lstm.StackState
        :type lengths: list of int
        :type actions: list of app.data.actions.action.Action
        """
        self.batch_size = batch_size
        self.hidden = hidden
        self.cell = cell
        self.previous = previous
        self.lengths = lengths
        self.actions = actions
