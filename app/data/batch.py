class Batch:

    def __init__(self, actions_tensor, actions_lengths, actions, tokens_tensor, tokens_lengths, tokens):
        """
        :type actions_tensor: torch.Tensor
        :type actions_lengths: list of int
        :type actions: list of list of app.data.actions.action.Action
        :type tokens_tensor: torch.Tensor
        :type tokens_lengths: list of int
        :type tokens: list of list of str
        """
        self.actions = BatchActions(actions_tensor, actions_lengths, actions)
        self.tokens = BatchTokens(tokens_tensor, tokens_lengths, tokens)
        self.max_actions_length = actions_tensor.size(0)
        self.max_tokens_length = tokens_tensor.size(0)
        self.size = actions_tensor.size(1)

    def get(self, index):
        """
        :type index: int
        :rtype: BatchElement
        """
        actions_tensor = self.actions.tensor[:, index:index+1]
        actions_length = self.actions.lengths[index]
        actions = self.actions.actions[index]
        tokens_tensor = self.tokens.tensor[:, index:index+1]
        tokens_length = self.tokens.lengths[index]
        tokens = self.tokens.tokens[index]
        return BatchElement(
            index,
            actions_tensor, actions_length, actions, self.max_actions_length,
            tokens_tensor, tokens_length, tokens, self.max_tokens_length
        )

class BatchActions:

    def __init__(self, tensor, lengths, actions):
        """
        :type tensor: torch.Tensor
        :type lengths: list of int
        :type actions: list of list of app.data.actions.action.Action
        """
        self.tensor = tensor
        self.lengths = lengths
        self.actions = actions

class BatchTokens:

    def __init__(self, tensor, lengths, tokens):
        """
        :type tensor: torch.Tensor
        :type lengths: list of int
        :type tokens: list of list of str
        """
        self.tensor = tensor
        self.lengths = lengths
        self.tokens = tokens

class BatchElement:

    def __init__(self,
        index,
        actions_tensor, actions_length, actions, max_actions_length,
        tokens_tensor, tokens_length, tokens, max_tokens_length
    ):
        """
        :type index: int
        :type actions_tensor: torch.Tensor
        :type actions_length: int
        :type actions: list of app.data.actions.action.Action
        :type max_actions_length: int
        :type tokens_tensor: torch.Tensor
        :type tokens_length: int
        :type tokens: list of str
        :type max_tokens_length: int
        """
        self.index = index
        self.actions = BatchElementActions(actions_tensor, actions_length, actions, max_actions_length)
        self.tokens = BatchElementTokens(tokens_tensor, tokens_length, tokens, max_tokens_length)

class BatchElementActions:

    def __init__(self, tensor, length, actions, max_length):
        """
        :type tensor: torch.Tensor
        :type length: int
        :type actions: list of app.data.actions.action.Action
        :type max_length: int
        """
        self.tensor = tensor
        self.length = length
        self.actions = actions
        self.max_length = max_length

class BatchElementTokens:

    def __init__(self, tensor, length, tokens, max_length):
        """
        :type tensor: torch.Tensor
        :type length: int
        :type tokens: list of str
        :type max_length: int
        """
        self.tensor = tensor
        self.length = length
        self.tokens = tokens
        self.max_length = max_length
