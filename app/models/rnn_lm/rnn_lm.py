from app.constants import PAD_INDEX
from app.data.action_sets.generative import GenerativeActionSet
from app.models.model import Model
from app.models.rnn_lm.state import StateFactory
import torch
from torch import nn

class RNNLM(Model):

    def __init__(self, device, rnn, action_embedding, action_embedding_size, action_converter):
        """
        :type device: torch.device
        :type rnn: app.rnn.rnn.RNN
        :type action_embedding: app.embeddings.embedding.Embedding
        :type action_embedding_size: int
        :type action_converter: app.data.converters.action.ActionConvter
        """
        super().__init__()
        self.device = device
        self.action_converter = action_converter
        self.action_count = self.action_converter.count()
        self.rnn = rnn
        rnn_output_size = rnn.get_output_size()
        self.rnn2logits = nn.Linear(in_features=rnn_output_size, out_features=self.action_count)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.action_embedding = action_embedding
        start_action_embedding = torch.FloatTensor(action_embedding_size).normal_()
        self.start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        self.action_set = GenerativeActionSet()
        self.state_factory = StateFactory(self.action_converter)

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor, dict
        """
        self.reset()
        start_action_embedding = self.start_action_embedding.view(1, 1, -1).expand(1, batch.size, -1)
        actions_tensor = batch.actions.tensor[:batch.max_actions_length - 1] # do not include last action
        actions_embeddings = self.action_embedding(actions_tensor)
        actions_embeddings = torch.cat((start_action_embedding, actions_embeddings), dim=0)
        initial_rnn_state = self.rnn.initial_state(batch.size)
        rnn_output, _ = self.rnn(actions_embeddings, initial_rnn_state)
        logits = self.rnn2logits(rnn_output)
        log_probs = self.log_softmax(logits)
        return log_probs, {}

    def initial_state(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, lengths):
        """
        Get initial state of model in a parse.

        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: app.models.rnn_lm.state.State
        """
        self.reset()
        batch_size = len(tokens)
        start_action_embedding = self.start_action_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        initial_rnn_state = self.rnn.initial_state(batch_size)
        _, rnn_state = self.rnn(start_action_embedding, initial_rnn_state)
        state = self.state_factory.initialize(rnn_state, lengths)
        return state

    def next_state(self, state, actions):
        """
        Advance state of the model to the next state.

        :type state: app.models.rnn_lm.state.State
        :type actions: list of app.data.actions.action.Action
        :rtype: app.models.rnn_lm.state.State
        """
        actions_tensor = self.actions2tensor(actions)
        actions_embedding = self.action_embedding(actions_tensor)
        _, rnn_state = self.rnn(actions_embedding, state.rnn_state)
        next_state = self.state_factory.next(rnn_state, actions)
        return next_state

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.rnn_lm.state.State
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor
        """
        rnn_output = self.rnn.state2output(state.rnn_state)
        logits = self.rnn2logits(rnn_output)
        log_probs = self.log_softmax(logits)
        return log_probs

    def valid_actions(self, state):
        """
        :type state: app.models.rnn_lm.state.State
        :rtype: list of list of int
        """
        iterator = zip(state.tokens_lengths, state.token_counter, state.last_action, state.open_nt_count)
        batch_valid_actions = []
        for tokens_length, token_counter, last_action, open_nt_count in iterator:
            valid_actions = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_nt_count)
            batch_valid_actions.append(valid_actions)
        return batch_valid_actions

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

    def reset(self):
        self.action_embedding.reset()
        self.rnn.reset()

    def actions2tensor(self, actions):
        """
        :type actions: list of app.data.actions.action.Action
        :rtype: torch.Tensor
        """
        indices = list(map(self.action2integer, actions))
        tensor = torch.tensor(indices, device=self.device, dtype=torch.long).unsqueeze(dim=0)
        return tensor

    def action2integer(self, action):
        """
        :type action:app.data.actions.action.Action
        :rtype: int
        """
        if action is None:
            return PAD_INDEX
        else:
            return self.action_converter.action2integer(action)

    def __str__(self):
        return (
            'RNNLM(\n'
            + f'  action_embedding={self.action_embedding}\n'
            + f'  rnn={self.rnn}\n'
            + ')'
        )
