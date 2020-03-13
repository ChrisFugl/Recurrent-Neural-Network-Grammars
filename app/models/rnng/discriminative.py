from app.constants import ACTION_SHIFT_INDEX, START_TOKEN_INDEX, START_TAG_INDEX
from app.data.action_set.discriminative import Discriminative as DiscriminativeActionSet
from app.models.rnng.rnng import RNNG
import torch
from torch import nn

class DiscriminativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, rnn_input_size, rnn_size, threads, token_size, pos_size, pos_embedding):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.stack.Stack, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type rnn_input_size: int
        :type rnn_size: int
        :type threads: int
        :type token_size: int
        :type pos_size: int
        :type pos_embedding: torch.nn.Embedding
        """
        super().__init__(device, embeddings, structures, converters, representation, composer, rnn_input_size, rnn_size, threads)
        self._action_set = DiscriminativeActionSet()
        self._generative = False
        self._pos_embedding = pos_embedding
        self._activation = nn.ReLU()
        self._word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)

    def _shift(self, log_probs, outputs, action):
        word_embedding = outputs.token_top.data
        token_top = self._token_buffer.pop(outputs.token_top)
        stack_top = self._stack.push(word_embedding, data=action, top=outputs.stack_top)
        action_log_prob = self._get_base_log_prop(log_probs, ACTION_SHIFT_INDEX)
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, token_top=token_top, token_counter=token_counter)

    def _initialize_token_buffer(self, tokens_tensor, tags_tensor, length):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.rnng.stack.StackNode
        """
        start_token_tensor = self._index2tensor(START_TOKEN_INDEX)
        start_tag_tensor = self._index2tensor(START_TAG_INDEX)
        start_word_embedding = self._get_word_embedding(start_token_tensor, start_tag_tensor)
        token_top = self._token_buffer.push(start_word_embedding, data=start_word_embedding)
        # discriminative model processes tokens in reverse order
        word_embeddings = self._get_word_embedding(tokens_tensor, tags_tensor)
        for index in range(length - 1, -1, -1):
            word_embedding = word_embeddings[index:index+1, :, :]
            token_top = self._token_buffer.push(word_embedding, data=word_embedding, top=token_top)
        return token_top

    def _get_word_embedding(self, token, tag):
        token_embedding = self._token_embedding(token)
        tag_embedding = self._pos_embedding(tag)
        word_embedding = torch.cat((token_embedding, tag_embedding), dim=2)
        word_embedding = self._activation(self._word2buffer(word_embedding))
        return word_embedding

    def __str__(self):
        return (
            'DiscriminativeRNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + ')'
        )
