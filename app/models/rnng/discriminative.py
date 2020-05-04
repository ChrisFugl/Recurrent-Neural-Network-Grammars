from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.rnng.rnng import RNNG
from app.utils import padded_reverse
from random import random
import torch
from torch import nn

class DiscriminativeRNNG(RNNG):

    def __init__(self,
        device, embeddings, structures, converters,
        representation, composer, sizes, threads,
        pos_size, pos_embedding, unk_token_prob, pretrained
    ):
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
        :type pretrained: app.embeddings.pretrained.PretrainedEmbedding
        """
        action_set = DiscriminativeActionSet()
        generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, threads, action_set, generative)
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

    def shift(self, log_probs, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, outputs, action):
        buffer_state = outputs.buffer_state
        stack_top = outputs.stack_top
        if self.uses_buffer or self.uses_stack:
            if self.uses_buffer:
                buffer_state = self.token_buffer.pop(buffer_state)
            if self.uses_stack:
                word_index = tokens_tensor.size(0) - outputs.token_counter - 1
                token_tensor = unknownified_tokens_tensor[word_index]
                if self.training and singletons_tensor[word_index] and random() > self.unk_token_prob:
                    token_tensor = tokens_tensor[word_index]
                pretrained = self.get_pretrained_embedding([[tokens[word_index]]], reverse=False)
                token_tensor = token_tensor.unsqueeze(dim=0)
                tag_tensor = tags_tensor[word_index].unsqueeze(dim=0)
                word_embedding = self.get_word_embedding(pretrained, token_tensor, tag_tensor)
                stack_top = self.stack.push(word_embedding, data=action, top=stack_top)
        action_log_prob = self.get_base_log_prop(log_probs, self.shift_index)
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, buffer_state=buffer_state, token_counter=token_counter)

    def initialize_token_buffer(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, lengths):
        """
        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of app.models.rnng.buffer.BufferState
        """
        batch_size = lengths.size(0)
        if self.start_pretrained_embedding is not None:
            start_pretrained_embedding = self.start_pretrained_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        else:
            start_pretrained_embedding = None
        start_token_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_tag_embedding = self.start_tag_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        start_word_embedding = self.merge_embeddings(start_pretrained_embedding, start_token_embedding, start_tag_embedding)
        unknownified_tokens_tensor_reversed = padded_reverse(unknownified_tokens_tensor, lengths)
        if self.training:
            singletons_tensor_reversed = padded_reverse(singletons_tensor, lengths)
            unknown_prob_mask = torch.rand(singletons_tensor_reversed.shape, device=self.device) < self.unk_token_prob
            unknown_mask = singletons_tensor_reversed & unknown_prob_mask
            tokens_tensor_reversed = padded_reverse(tokens_tensor, lengths)
            tokens_tensor_masked = torch.empty_like(tokens_tensor_reversed)
            tokens_tensor_masked.copy_(tokens_tensor_reversed)
            tokens_tensor_masked[unknown_mask] = unknownified_tokens_tensor_reversed[unknown_mask]
        else:
            tokens_tensor_masked = unknownified_tokens_tensor_reversed
        pretrained = self.get_pretrained_embedding(tokens)
        tags_tensor_reversed = padded_reverse(tags_tensor, lengths)
        word_embeddings = self.get_word_embedding(pretrained, tokens_tensor_masked, tags_tensor_reversed)
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        start_indices = lengths
        # plus one to account for start embedding
        buffer_states = self.token_buffer.initialize(word_embeddings, lengths + 1, start_indices)
        return buffer_states

    def get_word_embedding(self, pretrained, token, tag):
        token_embedding = self.token_embedding(token)
        tag_embedding = self.pos_embedding(tag)
        return self.merge_embeddings(pretrained, token_embedding, tag_embedding)

    def merge_embeddings(self, pretrained, token, tag):
        if pretrained is None:
            word = torch.cat((token, tag), dim=2)
        else:
            word = torch.cat((pretrained, token, tag), dim=2)
        word = self.word2buffer(word)
        word = self.activation(word)
        return word

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
            'DiscriminativeRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + f'  action_embedding={self.action_embedding}\n'
            + f'  nt_embedding={self.nt_embedding}\n'
            + f'  nt_compose_embedding={self.nt_compose_embedding}\n'
            + f'  token_embedding={self.token_embedding}\n'
            + f'  pos_embedding={self.pos_embedding}\n'
            + ('' if self.pretrained is None else f'  pretrained={self.pretrained}\n')
            + ')'
        )
