from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
from app.utils import padded_reverse
import torch
from torch import nn

class DiscriminativeParallelRNNG(ParallelRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size, pos_size, pos_embedding, unk_token_prob):
        """
        :type device: torch.device
        :type embeddings: app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.input_buffer_lstm.InputBufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type sample_stack_size: int
        :type pos_size: int
        :type pos_embedding: app.embeddings.embedding.Embedding
        :type unk_token_prob: float
        """
        action_set = DiscriminativeActionSet()
        generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size, action_set, generative)
        self.pos_embedding = pos_embedding
        self.unk_token_prob = unk_token_prob
        self.activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self.word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)
        start_tag_embedding = torch.FloatTensor(pos_size).normal_()
        self.start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

    def initialize_token_buffer(self, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, token_lengths):
        """
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: int
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_tag_embedding = self.start_tag_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_word_embedding = self.token_tag2word(start_token_embedding, start_tag_embedding)
        if self.training:
            singletons_tensor_reversed = padded_reverse(singletons_tensor, token_lengths - 1)
            unknown_prob_mask = torch.rand(singletons_tensor_reversed.shape, device=self.device) < self.unk_token_prob
            unknown_mask = singletons_tensor_reversed & unknown_prob_mask
            unknownified_tokens_reversed = padded_reverse(unknownified_tokens_tensor, token_lengths - 1)
            tokens = padded_reverse(tokens_tensor, token_lengths - 1)
            tokens[unknown_mask] = unknownified_tokens_reversed[unknown_mask]
        else:
            tokens = padded_reverse(unknownified_tokens_tensor, token_lengths - 1)
        tags = padded_reverse(tags_tensor, token_lengths - 1)
        word_embeddings = self.token_tag2embedding(tokens, tags)
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        self.token_buffer.initialize(word_embeddings, token_lengths)

    def get_word_embedding(self, state):
        """
        :type state: app.models.parallel_rnng.state.State
        :rtype: torch.Tensor, torch.Tensor
        """
        if self.training:
            unknown_prob_mask = torch.rand(state.singletons.shape, device=self.device) < self.unk_token_prob
            unknown_mask = state.singletons & unknown_prob_mask
            tokens = state.token_index
            tokens[unknown_mask] = state.unk_token_index[unknown_mask]
        else:
            tokens = state.unk_token_index
        word_embeddings = self.token_tag2embedding(tokens, state.tag_index, dim=1)
        return word_embeddings

    def update_token_buffer(self, batch_size, token_action_indices, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type word_embeddings: torch.Tensor
        """
        op = self.hold_op(batch_size)
        op[token_action_indices] = -1
        self.token_buffer.hold_or_pop(op)

    def token_tag2embedding(self, tokens, tags, dim=2):
        tokens_embedding = self.token_embedding(tokens)
        tags_embedding = self.pos_embedding(tags)
        return self.token_tag2word(tokens_embedding, tags_embedding, dim=dim)

    def token_tag2word(self, tokens, tags, dim=2):
        words = torch.cat((tokens, tags), dim=dim)
        words = self.word2buffer(words)
        words = self.activation(words)
        return words

    def __str__(self):
        return (
            'DiscriminativeParallelRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
