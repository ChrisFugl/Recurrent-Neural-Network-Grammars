"""
Modified from https://github.com/pytorch/pytorch/blob/cbcb2b5ad767622cf5ec04263018609bde3c974a/benchmarks/fastrnns/custom_lstms.py
"""
from app.dropout.variational import variational_dropout_mask
from app.rnn.rnn import RNN
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple

class LayerNormLSTM(RNN):

    def __init__(self, device, input_size, hidden_size, num_layers, dropout, weight_drop, dropout_type):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type dropout: float
        :type weight_drop: float
        :type dropout_type: str
        """
        super().__init__()
        self.device = device
        first_layer_args = [input_size, hidden_size, weight_drop]
        other_layer_args = [hidden_size, hidden_size, weight_drop]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.weight_drop = weight_drop
        self.use_weight_drop = weight_drop is not None
        self.lstm = StackedLSTM(
            device, num_layers, dropout, dropout_type, self.use_weight_drop, hidden_size,
            first_layer_args=first_layer_args, other_layer_args=other_layer_args
        )

    def forward(self, input, hidden_state):
        """
        :type input: torch.Tensor
        :type hidden_state: torch.Tensor, torch.Tensor
        :rtype: torch.Tensor, list of (torch.Tensor, torch.Tensor)
        """
        return self.lstm(input, hidden_state)

    def initial_state(self, batch_size):
        """
        Get initial hidden state.

        :type batch_size: int
        :rtype: list of (torch.Tensor, torch.Tensor)
        """
        shape = (batch_size, self.hidden_size)
        initial_states = []
        for _ in range(self.num_layers):
            cell = torch.zeros(shape, device=self.device, requires_grad=True)
            hidden = torch.zeros(shape, device=self.device, requires_grad=True)
            initial_states.append((hidden, cell))
        return initial_states

    def reset(self, batch_size):
        self.lstm.reset(batch_size)

    def get_output_size(self):
        """
        :rtype: int
        """
        return self.hidden_size

    def state2output(self, state):
        """
        :type state: torch.Tensor, torch.Tensor
        :rtype: torch.Tensor
        """
        last_layer = state[-1]
        hidden, _ = last_layer # (batch, hidden_size)
        output = hidden.unsqueeze(dim=0)
        return output

    def __str__(self):
        base_args = f'input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}'
        if self.use_weight_drop:
            if self.dropout is not None and self.dropout_type == 'variational':
                return f'LayerNormLSTM({base_args}, weight_drop={self.weight_drop}, dropout={self.dropout}, dropout_type={self.dropout_type})'
            else:
                return f'LayerNormLSTM({base_args}, weight_drop={self.weight_drop})'
        elif self.dropout is not None:
            return f'LayerNormLSTM({base_args}, dropout={self.dropout}, dropout_type={self.dropout_type})'
        else:
            return f'LayerNormLSTM({base_args})'

class LayerNormLSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, weight_drop):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / (4 * hidden_size)
        weight_ih = torch.empty(4 * hidden_size, input_size).uniform_(-k, k)
        bias_ih = torch.empty(4 * hidden_size).uniform_(-k, k)
        weight_hh = torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k)
        bias_hh = torch.empty(4 * hidden_size).uniform_(-k, k)
        self.weight_ih = Parameter(weight_ih)
        self.bias_ih = Parameter(bias_ih)
        self.bias_hh = Parameter(bias_hh)
        self.layer_norm = nn.LayerNorm(4 * hidden_size)
        self.use_weight_drop = weight_drop is not None
        self.weight_drop = weight_drop
        if self.use_weight_drop:
            self.weight_hh_raw = Parameter(weight_hh)
            self.reset(1)
        else:
            self.weight_hh = Parameter(weight_hh)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        hgates = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        gates = self.layer_norm(igates + hgates)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

    def set_weights(self):
        self.weight_hh = nn.functional.dropout(self.weight_hh_raw, p=self.weight_drop, training=self.training)

    def reset(self, batch_size):
        if self.use_weight_drop:
            self.set_weights()

class LSTMLayer(jit.ScriptModule):

    def __init__(self, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = LayerNormLSTMCell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

    def reset(self, batch_size):
        self.cell.reset(batch_size)

def init_stacked_lstm(num_layers, first_layer_args, other_layer_args):
    layers = [LSTMLayer(*first_layer_args)] + [LSTMLayer(*other_layer_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedLSTM(jit.ScriptModule):
    # Necessary for dropout support
    __constants__ = ['num_layers']

    def __init__(self, device, num_layers, dropout, dropout_type, use_weight_drop, hidden_size, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.device = device
        self.layers = init_stacked_lstm(num_layers, first_layer_args, other_layer_args)
        self.num_layers = num_layers
        self.use_dropout = dropout is not None
        dropout_p = 0.0 if dropout is None else dropout
        self.dropout_layer = nn.Dropout(dropout_p)
        self.dropout = dropout
        self.use_weight_drop = use_weight_drop
        self.use_variational = self.use_dropout and dropout_type == 'variational'
        self.use_normal = self.use_dropout and not use_weight_drop and dropout_type == 'normal'
        self.hidden_size = hidden_size
        # must declare masks to avoid compile error
        self.output_masks = [torch.empty((1))]
        self.hidden_masks = [torch.empty((1))]
        self.cell_masks = [torch.empty((1))]

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            if self.use_normal and i < self.num_layers - 1:
                output = self.dropout_layer(output)
            elif self.use_variational and self.training:
                batch_size = output.size(1)
                output = output * self.output_masks[i].expand(batch_size, -1)
                if not self.use_weight_drop:
                    hidden = out_state[0] * self.hidden_masks[i].expand(batch_size, -1)
                    cell = out_state[1] * self.cell_masks[i].expand(batch_size, -1)
                    out_state = hidden, cell
            output_states += [out_state]
            i += 1
        return output, output_states

    def reset(self, batch_size):
        for i in range(self.num_layers):
            self.layers[i].reset(batch_size)
        if self.use_variational and self.training:
            output_masks = []
            for _ in range(self.num_layers + 1):
                output_mask = variational_dropout_mask((1, self.hidden_size), self.dropout, device=self.device)
                output_masks.append(output_mask)
            self.output_masks = output_masks
            if not self.use_weight_drop:
                hidden_masks = []
                cell_masks = []
                for _ in range(self.num_layers + 1):
                    hidden_mask = variational_dropout_mask((1, self.hidden_size), self.dropout, device=self.device)
                    cell_mask = variational_dropout_mask((1, self.hidden_size), self.dropout, device=self.device)
                    hidden_masks.append(hidden_mask)
                    cell_masks.append(cell_mask)
                self.hidden_masks = hidden_masks
                self.cell_masks = cell_masks
