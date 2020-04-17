from app.models.model import Model
import torch
from torch import nn

class AbstractRNNG(Model):
    """
    Base class for RNNG and Parallel RNNG. This class makes it easy to ensure that parameters are setup the same way.
    """

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, action_set, generative, base_out_features):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: object, object, object
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type action_set: app.data.action_sets.action_set.ActionSet
        :type generative: bool
        :type base_out_features: int
        """
        super().__init__()
        self.action_set = action_set
        self.generative = generative
        action_size, token_size, rnn_input_size, rnn_size = sizes
        self.device = device
        self.action_converter, self.token_converter, self.tag_converter, self.non_terminal_converter = converters
        self.action_embedding, self.token_embedding, self.nt_embedding, self.nt_compose_embedding = embeddings
        self.representation = representation
        self.uses_buffer = representation.uses_token_buffer()
        self.uses_history = representation.uses_action_history()
        self.uses_stack = representation.uses_stack()
        action_history, token_buffer, stack = structures
        if self.uses_history:
            self.action_history = action_history
            start_action_embedding = torch.FloatTensor(action_size).uniform_(-1, 1)
            self.start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        if self.uses_buffer:
            self.token_buffer = token_buffer
            start_token_embedding = torch.FloatTensor(token_size).uniform_(-1, 1)
            self.start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)
        if self.uses_stack:
            self.stack = stack
            start_stack_embedding = torch.FloatTensor(rnn_input_size).uniform_(-1, 1)
            self.start_stack_embedding = nn.Parameter(start_stack_embedding, requires_grad=True)
        self.representation2logits = nn.Linear(in_features=rnn_input_size, out_features=base_out_features, bias=True)
        self.composer = composer
        self.logits2log_prob = nn.LogSoftmax(dim=2)

    def save(self, path):
        """
        Save model parameters.

        :type path: str
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        """
        Load model parameters from file.

        :type path: str
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
