from app.constants import PAD_INDEX
from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.model import Model
from app.models.rnn_parser.attention import Attention
from app.models.rnn_parser.decoder import Decoder
from app.models.rnn_parser.encoder import Encoder
from app.models.rnn_parser.state import StateFactory
from app.utils import padded_reverse
import torch
from torch import nn

class RNNParser(Model):

    def __init__(self, device, encoder_rnn, decoder_rnn, action_embedding, action_embedding_size, token_embedding, action_converter):
        """
        :type device: torch.device
        :type encoder_rnn: app.rnn.rnn.RNN
        :type decoder_rnn: app.rnn.rnn.RNN
        :type action_embedding: app.embeddings.embedding.Embedding
        :type token_embedding: app.embeddings.embedding.Embedding
        :type action_converter: app.data.converters.action.ActionConvter
        """
        super().__init__()
        encoder_output_size = encoder_rnn.get_output_size()
        decoder_output_size = decoder_rnn.get_output_size()
        self.device = device
        self.action_converter = action_converter
        self.action_count = self.action_converter.count()
        self.action_embedding = action_embedding
        self.token_embedding = token_embedding
        self.encoder = Encoder(encoder_rnn)
        self.decoder = Decoder(decoder_rnn, self.action_count)
        self.attention = Attention(encoder_output_size, decoder_output_size)
        start_action_embedding = torch.FloatTensor(action_embedding_size).normal_()
        self.start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        self.action_set = DiscriminativeActionSet()
        self.state_factory = StateFactory(self.action_converter)

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor, dict
        """
        self.reset()
        tokens_reversed = padded_reverse(batch.unknownified_tokens.tensor, batch.tokens.lengths)
        token_embeddings = self.token_embedding(tokens_reversed)
        encoder_outputs = self.encoder(token_embeddings)
        decoder_state = self.decoder.initial_hidden_state(batch.size)
        previous_action = self.start_action_embedding.view(1, 1, -1).expand(1, batch.size, -1)
        attention_weights = []
        outputs = torch.zeros((batch.max_actions_length, batch.size, self.action_count), device=self.device, dtype=torch.float)
        for i in range(batch.max_actions_length):
            decoder_rnn_output = self.decoder.state2tensor(decoder_state)
            attention_output, weights = self.attention(encoder_outputs, batch.tokens.lengths, decoder_rnn_output)
            attention_weights.append(weights.detach().cpu())
            log_probs, decoder_state = self.decoder(attention_output, previous_action, decoder_state)
            actions_tensor = batch.actions.tensor[i].unsqueeze(dim=0)
            previous_action = self.action_embedding(actions_tensor)
            outputs[i] = log_probs
        info = {'attention': attention_weights}
        return outputs, info

    def initial_state(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, lengths):
        """
        Get initial state of model in a parse.

        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: app.models.rnn_parser.state.State
        """
        self.reset()
        batch_size = len(tokens)
        tokens_reversed = padded_reverse(unknownified_tokens_tensor, lengths)
        token_embeddings = self.token_embedding(tokens_reversed)
        encoder_outputs = self.encoder(token_embeddings)
        decoder_state = self.decoder.initial_hidden_state(batch_size)
        previous_action = self.start_action_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        state = self.state_factory.initialize(encoder_outputs, previous_action, decoder_state, lengths)
        return state

    def next_state(self, state, actions):
        """
        Advance state of the model to the next state.

        :type state: app.models.rnn_parser.state.State
        :type actions: list of app.data.actions.action.Action
        :rtype: app.models.rnn_parser.state.State
        """
        actions_tensor = self.actions2tensor(actions)
        previous_action = self.action_embedding(actions_tensor)
        decoder_rnn_output = self.decoder.state2tensor(state.decoder_state)
        attention_output, _ = self.attention(state.encoder_outputs, state.tokens_lengths, decoder_rnn_output)
        _, decoder_state = self.decoder(attention_output, state.previous_action, state.decoder_state)
        next_state = self.state_factory.next(state, previous_action, decoder_state, actions)
        return next_state

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.rnn_parser.state.State
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor
        """
        decoder_rnn_output = self.decoder.state2tensor(state.decoder_state)
        attention_output, _ = self.attention(state.encoder_outputs, state.tokens_lengths, decoder_rnn_output)
        log_probs, _ = self.decoder(attention_output, state.previous_action, state.decoder_state)
        return log_probs

    def valid_actions(self, state):
        """
        :type state: app.models.rnn_parser.state.State
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
        self.encoder.reset()
        self.decoder.reset()
        self.action_embedding.reset()
        self.token_embedding.reset()

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
            'RNNParser(\n'
            + f'  action_embedding={self.action_embedding}\n'
            + f'  token_embedding={self.token_embedding}\n'
            + f'  encoder={self.encoder}\n'
            + f'  decoder={self.decoder}\n'
            + ')'
        )
