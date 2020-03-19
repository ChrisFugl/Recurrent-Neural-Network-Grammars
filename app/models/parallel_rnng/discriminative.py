from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
import torch
from torch import nn

class DiscriminativeParallelRNNG(ParallelRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, pos_size, pos_embedding):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.stack_lstm.StackLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type pos_size: int
        :type pos_embedding: torch.nn.Embedding
        """
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes)
        self._generative = False
        self._pos_embedding = pos_embedding
        self._activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self._word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)
        start_token_embedding = torch.FloatTensor(1, 1, token_size).uniform_(-1, 1)
        self._start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)
        start_tag_embedding = torch.FloatTensor(1, 1, pos_size).uniform_(-1, 1)
        self._start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

    def _initialize_token_buffer(self, tokens_tensor, tags_tensor, token_lengths):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self._batch_one_element_tensor(self._start_token_embedding, batch_size)
        start_tag_embedding = self._batch_one_element_tensor(self._start_tag_embedding, batch_size)
        start_word_embedding = self._token_tag2word(start_token_embedding, start_tag_embedding)
        # add tokens in reverse order
        word_embeddings = self._token_tag2embedding(tokens_tensor, tags_tensor).flip(dims=[0])
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        token_buffer = self._token_buffer.initialize(word_embeddings, token_lengths)
        return token_buffer

    def _get_word_embedding(self, preprocessed, token_action_indices):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type token_action_indices: torch.Tensor
        :rtype: torch.Tensor, torch.Tensor
        """
        token_indices = preprocessed.token_index[token_action_indices]
        tag_indices = preprocessed.tag_index[token_action_indices]
        word_embeddings = self._token_tag2embedding(token_indices, tag_indices, dim=1)
        return word_embeddings

    def _update_token_buffer(self, batch_size, token_action_indices, token_buffer, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type token_buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :type word_embeddings: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        op = self._hold_op(batch_size)
        op[token_action_indices] = -1
        token_buffer, _ = self._token_buffer.hold_or_pop(token_buffer, op)
        return token_buffer

    def _token_tag2embedding(self, tokens, tags, dim=2):
        tokens_embedding = self._token_embedding(tokens)
        tags_embedding = self._pos_embedding(tags)
        return self._token_tag2word(tokens_embedding, tags_embedding, dim=dim)

    def _token_tag2word(self, tokens, tags, dim=2):
        words = torch.cat((tokens, tags), dim=dim)
        words = self._activation(self._word2buffer(words))
        return words

    def __str__(self):
        return (
            'DiscriminativeParallelRNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + ')'
        )
