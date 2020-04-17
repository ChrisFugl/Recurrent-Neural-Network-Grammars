from app.dropout.weight_drop import WeightDrop
from torch import nn

class BufferLSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, weight_drop):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type weight_drop: float
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout)
        if weight_drop is not None:
            weights = [f'weight_hh_l{i}' for i in range(self.num_layers)]
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias)
            self.lstm = WeightDrop(self.lstm, weights, weight_drop)
