from torch import nn

class MemoryLSTM(nn.Module):
    """
    Base class for stack and history classes.
    """

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
        self._device = device
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            dropout=dropout,
        )
