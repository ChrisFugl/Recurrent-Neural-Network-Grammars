"""
Based on https://github.com/shuoyangd/hoolock/blob/master/model/MultiLayerLSTMCell.py
"""
from app.dropout.weight_drop import WeightDrop
import torch
from torch import nn

class MultiLayerLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, bias, dropout, weight_drop):
      """
      :type input_size: int
      :type hidden_size: int
      :type num_layers: int
      :type bias: bool
      :type dropout: float
      :type weight_drop: float
      """
      super(MultiLayerLSTMCell, self).__init__()
      self.num_layers = num_layers
      self.dropout = nn.Dropout(p=dropout)
      self.lstm = nn.ModuleList()
      self.lstm.append(nn.LSTMCell(input_size, hidden_size, bias=bias))
      weights = ['weight_hh']
      for i in range(num_layers - 1):
          self.lstm.append(nn.LSTMCell(hidden_size, hidden_size, bias=bias))
      self.use_weight_drop = weight_drop is not None
      if self.use_weight_drop:
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
          next_hidden_i, next_cell_i = self.lstm[i](lstm_input, (prev_hidden_i, prev_cell_i))
          next_hidden += [next_hidden_i]
          next_cell += [next_cell_i]
          if self.use_weight_drop:
              lstm_input = next_hidden_i
          else:
              lstm_input = self.dropout(next_hidden_i)
      next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
      next_cell = torch.stack(next_cell).permute(1, 2, 0)
      return next_hidden, next_cell


  def reset(self):
      if self.use_weight_drop:
          for i in range(self.num_layers):
              self.lstm[i].reset()
