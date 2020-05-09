"""
Based on https://github.com/shuoyangd/hoolock/blob/master/model/MultiLayerLSTMCell.py
"""
from app.dropout.variational import variational_dropout_mask
from app.dropout.weight_drop import WeightDrop
from app.rnn.layer_norm_lstm import LayerNormLSTMCell
import torch
from torch import nn

class MultiLayerLSTMCell(nn.Module):

  def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, weight_drop, dropout_type, layer_norm):
      """
      :type input_size: int
      :type hidden_size: int
      :type num_layers: int
      :type bias: bool
      :type dropout: float
      :type weight_drop: float
      """
      super(MultiLayerLSTMCell, self).__init__()
      self.device = device
      self.num_layers = num_layers
      self.dropout = dropout
      dropout_p = 0.0 if dropout is None else dropout
      self.dropout_layer = nn.Dropout(p=dropout_p)
      self.lstm = nn.ModuleList()
      self.layer_norm = layer_norm
      self.hidden_size = hidden_size
      self.use_dropout = dropout is not None
      self.use_weight_drop = weight_drop is not None
      self.use_variational = self.use_dropout and dropout_type == 'variational'
      self.use_normal = self.use_dropout and not self.use_weight_drop and dropout_type == 'normal'
      if layer_norm:
          self.lstm.append(LayerNormLSTMCell(input_size, hidden_size, weight_drop))
          for i in range(num_layers - 1):
              self.lstm.append(LayerNormLSTMCell(hidden_size, hidden_size, weight_drop))
      else:
          self.lstm.append(nn.LSTMCell(input_size, hidden_size, bias=bias))
          weights = ['weight_hh']
          for i in range(num_layers - 1):
              self.lstm.append(nn.LSTMCell(hidden_size, hidden_size, bias=bias))
          if self.use_weight_drop and not layer_norm:
              for i in range(num_layers):
                  self.lstm[i] = WeightDrop(self.lstm[i], weights, weight_drop)

  def forward(self, input, prev):
      """
      :param input: (batch_size, input_size)
      :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
      """
      next_hidden = []
      next_cell = []
      lstm_input = input
      for i in range(self.num_layers):
          prev_hidden_i = prev[0][:, :, i]
          prev_cell_i = prev[1][:, :, i]
          if self.layer_norm:
              _, (next_hidden_i, next_cell_i) = self.lstm[i](lstm_input, (prev_hidden_i, prev_cell_i))
          else:
              next_hidden_i, next_cell_i = self.lstm[i](lstm_input, (prev_hidden_i, prev_cell_i))
          lstm_input = next_hidden_i
          if self.use_normal:
              lstm_input = self.dropout_layer(next_hidden_i)
          elif self.use_variational and self.training:
              batch_size = input.size(0)
              lstm_input = next_hidden_i * self.output_masks[i].expand(batch_size, -1)
              if not self.use_weight_drop:
                  next_hidden_i = next_hidden_i * self.hidden_masks[i].expand(batch_size, -1)
                  next_cell_i = next_cell_i * self.hidden_masks[i].expand(batch_size, -1)
          next_hidden += [next_hidden_i]
          next_cell += [next_cell_i]
      next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
      next_cell = torch.stack(next_cell).permute(1, 2, 0)
      return next_hidden, next_cell

  def reset(self, batch_size):
      if self.use_weight_drop:
          for i in range(self.num_layers):
              self.lstm[i].reset(batch_size)
      if self.use_variational and self.training:
          self.output_masks = []
          for _ in range(self.num_layers):
              output_mask = variational_dropout_mask((batch_size, self.hidden_size), self.dropout, device=self.device)
              self.output_masks.append(output_mask)
          if not self.use_weight_drop:
              self.hidden_masks = []
              self.cell_masks = []
              for _ in range(self.num_layers):
                  hidden_mask = variational_dropout_mask((1, self.hidden_size), self.dropout, device=self.device)
                  cell_mask = variational_dropout_mask((1, self.hidden_size), self.dropout, device=self.device)
                  self.hidden_masks.append(hidden_mask)
                  self.cell_masks.append(cell_mask)
