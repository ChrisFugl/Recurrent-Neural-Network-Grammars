from torch import nn

class BufferLSTM(nn.Module):

    def __init__(self, device, rnn):
        """
        :type device: torch.device
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.device = device
        self.rnn = rnn

    def reset(self, batch_size):
        self.rnn.reset(batch_size)
