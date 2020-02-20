from torch import nn

class RNN(nn.Module):

    def forward(self, input, hidden_state):
        """
        :type input: torch.Tensor
        :type hidden_state: object
        :rtype: torch.Tensor, object
        """
        raise NotImplementedError('must be implemented by subclass')

    def initial_state(self):
        """
        Get initial hidden state.

        :rtype: object
        """
        raise NotImplementedError('must be implemented by subclass')
