class Batch:

    def __init__(self,
        actions_tensor, actions_lengths, actions,
        tokens_tensor, tokens_lengths, tokens,
        unknownified_tokens_tensor, unknownified_tokens,
        singletons,
        tags_tensor, tags
    ):
        """
        :type actions_tensor: torch.Tensor
        :type actions_lengths: torch.Tensor
        :type actions: list of list of app.data.actions.action.Action
        :type tokens_tensor: torch.Tensor
        :type tokens_lengths: torch.Tensor
        :type tokens: list of list of str
        :type unknownified_tokens_tensor: torch.Tensor
        :type unknownified_tokens: list of list of str
        :type singletons: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type tags: list of list of str
        """
        self.actions = BatchActions(actions_tensor, actions_lengths, actions)
        self.tokens = BatchTokens(tokens_tensor, tokens_lengths, tokens)
        self.unknownified_tokens = BatchTokens(unknownified_tokens_tensor, tokens_lengths, unknownified_tokens)
        self.singletons = singletons
        self.tags = BatchTags(tags_tensor, tags)
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
        unknownified_tokens_tensor = self.unknownified_tokens.tensor[:, index:index+1]
        unknownified_tokens = self.unknownified_tokens.tokens[index]
        singletons = self.singletons[:, index:index+1]
        tags_tensor = self.tags.tensor[:, index:index+1]
        tags = self.tags.tags[index]
        return BatchElement(
            index,
            actions_tensor, actions_length, actions, self.max_actions_length,
            tokens_tensor, tokens_length, tokens, self.max_tokens_length,
            unknownified_tokens_tensor, unknownified_tokens,
            singletons,
            tags_tensor, tags
        )

class BatchActions:

    def __init__(self, tensor, lengths, actions):
        """
        :type tensor: torch.Tensor
        :type lengths: torch.Tensor
        :type actions: list of list of app.data.actions.action.Action
        """
        self.tensor = tensor
        self.lengths = lengths
        self.actions = actions

class BatchTokens:

    def __init__(self, tensor, lengths, tokens):
        """
        :type tensor: torch.Tensor
        :type lengths: torch.Tensor
        :type tokens: list of list of str
        """
        self.tensor = tensor
        self.lengths = lengths
        self.tokens = tokens

class BatchTags:

    def __init__(self, tensor, tags):
        """
        :type tensor: torch.Tensor
        :type tags: list of list of str
        """
        self.tensor = tensor
        self.tags = tags

class BatchElement:

    def __init__(self,
        index,
        actions_tensor, actions_length, actions, max_actions_length,
        tokens_tensor, tokens_length, tokens, max_tokens_length,
        unknownified_tokens_tensor, unknownified_tokens,
        singletons,
        tags_tensor, tags
    ):
        """
        :type index: int
        :type actions_tensor: torch.Tensor
        :type actions_length: torch.Tensor
        :type actions: list of app.data.actions.action.Action
        :type max_actions_length: int
        :type tokens_tensor: torch.Tensor
        :type tokens_length: torch.Tensor
        :type tokens: list of str
        :type max_tokens_length: int
        :type unknownified_tokens_tensor: torch.Tensor
        :type unknownified_tokens: list of str
        :type singletons: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type tags: list of str
        """
        self.index = index
        self.actions = BatchElementActions(actions_tensor, actions_length, actions, max_actions_length)
        self.tokens = BatchElementTokens(tokens_tensor, tokens_length, tokens, max_tokens_length)
        self.unknownified_tokens = BatchElementTokens(unknownified_tokens_tensor, tokens_length, unknownified_tokens, max_tokens_length)
        self.singletons = singletons
        self.tags = BatchElementTags(tags_tensor, tags)

class BatchElementActions:

    def __init__(self, tensor, length, actions, max_length):
        """
        :type tensor: torch.Tensor
        :type length: torch.Tensor
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
        :type length: torch.Tensor
        :type tokens: list of str
        :type max_length: int
        """
        self.tensor = tensor
        self.length = length
        self.tokens = tokens
        self.max_length = max_length

class BatchElementTags:

    def __init__(self, tensor, tags):
        """
        :type tensor: torch.Tensor
        :type tags: list of str
        """
        self.tensor = tensor
        self.tags = tags
