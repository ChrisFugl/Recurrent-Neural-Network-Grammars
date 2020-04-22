import torch
from torch import nn

class Stack(nn.Module):

    def __init__(self, rnn):
        """
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.rnn = rnn

    def contents(self, top):
        """
        :rtype: torch.Tensor
        :returns: sequence of all states in stack
        """
        embeddings = []
        node = top
        while node is not None:
            embeddings.append(node.output)
            node = node.parent
        embeddings.reverse()
        return torch.cat(embeddings, dim=0)

    def push(self, embedding, data=None, top=None):
        """
        :type embedding: torch.Tensor
        :type data: object
        :type top: StackNode
        :rtype: StackNode
        :returns: the stack state after push
        """
        if top is None:
            _, batch_size, _ = embedding.shape
            state = self.rnn.initial_state(batch_size)
        else:
            state = top.state
        output, next_state = self.rnn(embedding, state)
        next_top = StackNode(output, next_state, top, data)
        return next_top

    def pop(self, top):
        """
        :type top: StackNode
        :rtype: StackNode
        :returns: state of the stack after pop, data
        """
        if top is None:
            raise Exception('Pop operation is impossible on an empty stack.')
        else:
            return top.parent

    def top(self, top):
        """
        :type top: StackNode
        :rtype: torch.Tensor
        """
        return top.output

    def __str__(self):
        return f'Stack(rnn={self.rnn})'

class StackNode:

    def __init__(self, output, state, parent, data):
        self.output = output
        self.state = state
        self.parent = parent
        self.data = data
        if parent is None:
            self.length = 1
        else:
            self.length = parent.length + 1

    def length_as_tensor(self, device):
        """
        :type device: torch.device
        :rtype: torch.Tensor
        """
        return torch.tensor([self.length], device=device, dtype=torch.long)
