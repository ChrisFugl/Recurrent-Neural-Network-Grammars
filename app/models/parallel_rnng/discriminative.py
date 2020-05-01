from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
from app.utils import padded_reverse
import torch
from torch import nn

class DiscriminativeParallelRNNG(ParallelRNNG):

    def __init__(self,
        device, embeddings, structures, converters,
        representation, composer,
        sizes, sample_stack_size, pos_size,
        pos_embedding, unk_token_prob, pretrained
    ):
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
        :type pretrained: app.embeddings.pretrained.PretrainedEmbedding
        """
        action_set = DiscriminativeActionSet()
        generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size, action_set, generative)
        self.pos_embedding = pos_embedding
        self.unk_token_prob = unk_token_prob
        self.pretrained = pretrained
        self.activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self.start_pretrained_embedding = None
        word2buffer_input_size = token_size + pos_size
        if pretrained is not None:
            word2buffer_input_size += pretrained.size()
            start_pretrained_embedding = torch.FloatTensor(pretrained.size()).normal_()
            self.start_pretrained_embedding = nn.Parameter(start_pretrained_embedding)
        self.word2buffer = nn.Linear(in_features=word2buffer_input_size, out_features=rnn_input_size, bias=True)
        start_tag_embedding = torch.FloatTensor(pos_size).normal_()
        self.start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

    def initialize_token_buffer(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, token_lengths):
        """
        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: int
        """
        batch_size = tokens_tensor.size(1)
        if self.start_pretrained_embedding is not None:
            start_pretrained_embedding = self.start_pretrained_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        else:
            start_pretrained_embedding = None
        start_token_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_tag_embedding = self.start_tag_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_word_embedding = self.merge_embeddings(start_pretrained_embedding, start_token_embedding, start_tag_embedding)
        unknownified_tokens_tensor_reversed = padded_reverse(unknownified_tokens_tensor, token_lengths - 1)
        if self.training:
            singletons_tensor_reversed = padded_reverse(singletons_tensor, token_lengths - 1)
            unknown_prob_mask = torch.rand(singletons_tensor_reversed.shape, device=self.device) < self.unk_token_prob
            unknown_mask = singletons_tensor_reversed & unknown_prob_mask
            tokens_tensor_reversed = padded_reverse(tokens_tensor, token_lengths - 1)
            tokens_tensor_masked = torch.empty_like(tokens_tensor_reversed)
            tokens_tensor_masked.copy_(tokens_tensor_reversed)
            tokens_tensor_masked[unknown_mask] = unknownified_tokens_tensor_reversed[unknown_mask]
        else:
            tokens_tensor_masked = unknownified_tokens_tensor_reversed
        pretrained = self.get_pretrained_embedding(tokens)
        tags_tensor_reversed = padded_reverse(tags_tensor, token_lengths - 1)
        word_embeddings = self.inputs2embedding(pretrained, tokens_tensor_masked, tags_tensor_reversed)
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
        pretrained = self.get_pretrained_embedding([state.tokens], reverse=False)
        if pretrained is not None:
            pretrained = pretrained.view(len(state.tokens), -1)
        word_embeddings = self.inputs2embedding(pretrained, tokens, state.tag_index, dim=1)
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

    def inputs2embedding(self, pretrained_embedding, tokens, tags, dim=2):
        tokens_embedding = self.token_embedding(tokens)
        tags_embedding = self.pos_embedding(tags)
        return self.merge_embeddings(pretrained_embedding, tokens_embedding, tags_embedding, dim=dim)

    def merge_embeddings(self, pretrained, tokens, tags, dim=2):
        if pretrained is None:
            words = torch.cat((tokens, tags), dim=dim)
        else:
            words = torch.cat((pretrained, tokens, tags), dim=dim)
        words = self.word2buffer(words)
        words = self.activation(words)
        return words

    def get_pretrained_embedding(self, tokens, reverse=True):
        if self.pretrained is None:
            return None
        else:
            if reverse:
                tokens = self.reverse_tokens(tokens)
            return self.pretrained(tokens)

    def reverse_tokens(self, tokens):
        reversed_tokens = []
        for sentence in tokens:
            reversed_tokens.append(list(reversed(sentence)))
        return reversed_tokens

    def __str__(self):
        return (
            'DiscriminativeParallelRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ('' if self.pretrained is None else f'  pretrained={self.pretrained}\n')
            + ')'
        )
