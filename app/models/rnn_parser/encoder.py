from torch import nn

class Encoder(nn.Module):

    def __init__(self, rnn):
        """
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.rnn = rnn

    def forward(self, tokens_reversed):
        """
        :type tokens_reversed: torch.Tensor
        :rtype: torch.Tensor
        """
        batch_size = tokens_reversed.size(1)
        initial_state = self.rnn.initial_state(batch_size)
        output, _ = self.rnn(tokens_reversed, initial_state)
        return output

    def reset(self):
        self.rnn.reset()

    def __str__(self):
        return f'Encoder(rnn={self.rnn})'
