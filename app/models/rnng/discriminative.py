from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.rnng.rnng import RNNG
from app.utils import padded_reverse
from random import random
import torch
from torch import nn

class DiscriminativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads, pos_size, pos_embedding, unk_token_prob):
        """
        :type device: torch.device
        :type embeddings: app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.buffer.Buffer, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type threads: int
        :type pos_size: int
        :type pos_embedding: app.embeddings.embedding.Embedding
        :type unk_token_prob: float
        """
        action_set = DiscriminativeActionSet()
        generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, threads, action_set, generative)
        self.pos_embedding = pos_embedding
        self.unk_token_prob = unk_token_prob
        self.activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self.word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)
        start_tag_embedding = torch.FloatTensor(pos_size).normal_()
        self.start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

    def shift(self, log_probs, tokens, unknownified_tokens, singletons, tags, outputs, action):
        buffer_state = outputs.buffer_state
        stack_top = outputs.stack_top
        if self.uses_buffer or self.uses_stack:
            if self.uses_buffer:
                buffer_state = self.token_buffer.pop(buffer_state)
            if self.uses_stack:
                word_index = tokens.size(0) - outputs.token_counter - 1
                token = unknownified_tokens[word_index]
                if self.training and singletons[word_index] and random() > self.unk_token_prob:
                    token = tokens[word_index]
                token = token.unsqueeze(dim=0)
                tag = tags[word_index].unsqueeze(dim=0)
                word_embedding = self.get_word_embedding(token, tag)
                stack_top = self.stack.push(word_embedding, data=action, top=stack_top)
        action_log_prob = self.get_base_log_prop(log_probs, self.shift_index)
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, buffer_state=buffer_state, token_counter=token_counter)

    def initialize_token_buffer(self, tokens, unknownified_tokens, singletons, tags, lengths):
        """
        :type tokens: torch.Tensor
        :type unknownified_tokens: torch.Tensor
        :type singletons: torch.Tensor
        :type tags: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of app.models.rnng.buffer.BufferState
        """
        batch_size = lengths.size(0)
        start_token_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_tag_embedding = self.start_tag_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_word_embedding = self.token_tag2word(start_token_embedding, start_tag_embedding)
        if self.training:
            singletons_reversed = padded_reverse(singletons, lengths)
            unknownified_tokens_reversed = padded_reverse(unknownified_tokens, lengths)
            tokens_reversed = padded_reverse(tokens, lengths)
            unknown_mask = singletons_reversed & (torch.rand(singletons.shape, device=self.device) < self.unk_token_prob)
            tokens_reversed[unknown_mask] = unknownified_tokens_reversed[unknown_mask]
        else:
            tokens_reversed = padded_reverse(unknownified_tokens, lengths)
        tags_reversed = padded_reverse(tags, lengths)
        word_embeddings = self.get_word_embedding(tokens_reversed, tags_reversed)
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        start_indices = lengths
        # plus one to account for start embedding
        buffer_states = self.token_buffer.initialize(word_embeddings, lengths + 1, start_indices)
        return buffer_states

    def get_word_embedding(self, token, tag):
        token_embedding = self.token_embedding(token)
        tag_embedding = self.pos_embedding(tag)
        return self.token_tag2word(token_embedding, tag_embedding)

    def token_tag2word(self, token, tag):
        word = torch.cat((token, tag), dim=2)
        word = self.word2buffer(word)
        word = self.activation(word)
        return word

    def __str__(self):
        return (
            'DiscriminativeRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
