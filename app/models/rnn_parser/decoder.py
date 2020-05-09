import torch
from torch import nn

class Decoder(nn.Module):

    def __init__(self, rnn, action_count):
        """
        :type rnn: app.rnn.rnn.RNN
        :type action_count: int
        """
        super().__init__()
        self.rnn = rnn
        self.rnn_output_size = rnn.get_output_size()
        self.rnn2logits = nn.Linear(in_features=self.rnn_output_size, out_features=action_count)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, attention_output, previous_action, decoder_state):
        """
        :param attention_output: 1 x batch size x encoder output size
        :type attention_output: torch.Tensor
        :param previous_action: tensor, 1 x batch size x action embedding size
        :type previous_action: torch.Tensor
        :returns: output (tensor, batch size x action count), rnn state
        """
        rnn_input = torch.cat((attention_output, previous_action), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, decoder_state)
        batch_size = rnn_output.size(1)
        output = rnn_output.view(batch_size, self.rnn_output_size)
        output = self.rnn2logits(output)
        output = self.log_softmax(output)
        return output, rnn_state

    def state2tensor(self, state):
        """
        :rtype: torch.Tensor
        """
        return self.rnn.state2output(state)

    def reset(self, batch_size):
        self.rnn.reset(batch_size)

    def __str__(self):
        return f'Decoder(rnn={self.rnn})'
