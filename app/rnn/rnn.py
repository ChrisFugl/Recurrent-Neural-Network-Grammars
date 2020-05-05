from torch import nn

class RNN(nn.Module):

    def forward(self, input, hidden_state):
        """
        :type input: torch.Tensor
        :type hidden_state: object
        :rtype: torch.Tensor, object
        """
        raise NotImplementedError('must be implemented by subclass')

    def initial_state(self, batch_size):
        """
        Get initial hidden state.

        :type batch_size: int
        :rtype: object
        """
        raise NotImplementedError('must be implemented by subclass')

    def reset(self):
        """
        Resets internal state.
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_output_size(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def state2output(self, state):
        """
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
