from app.data.action_set.discriminative import Discriminative
from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
import torch
from torch import nn

class DiscriminativeParallelRNNG(ParallelRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, pos_size, pos_embedding):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.input_buffer_lstm.InputBufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type pos_size: int
        :type pos_embedding: torch.nn.Embedding
        """
        self.generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes)
        self.action_set = Discriminative()
        self.pos_embedding = pos_embedding
        self.activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self.word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)
        start_token_embedding = torch.FloatTensor(1, token_size).uniform_(-1, 1)
        self.start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)
        start_tag_embedding = torch.FloatTensor(1, pos_size).uniform_(-1, 1)
        self.start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

        self.reduce_index = self.action_converter.string2integer('REDUCE')
        self.shift_index = self.action_converter.string2integer('SHIFT')
        self.nt_start_index = self.action_converter.get_non_terminal_offset()

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, token_lengths):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self.batch_one_element_tensor(self.start_token_embedding, batch_size).unsqueeze(dim=0)
        start_tag_embedding = self.batch_one_element_tensor(self.start_tag_embedding, batch_size).unsqueeze(dim=0)
        start_word_embedding = self.token_tag2word(start_token_embedding, start_tag_embedding)
        # add tokens in reverse order
        word_embeddings = self.token_tag2embedding(tokens_tensor, tags_tensor).flip(dims=[0])
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        self.token_buffer.initialize(word_embeddings, token_lengths)

    def get_word_embedding(self, preprocessed, token_action_indices):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type token_action_indices: torch.Tensor
        :rtype: torch.Tensor, torch.Tensor
        """
        token_indices = preprocessed.token_index
        tag_indices = preprocessed.tag_index
        word_embeddings = self.token_tag2embedding(token_indices, tag_indices, dim=1)
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
        words = self.activation(self.word2buffer(words))
        return words

    def inference_initialize_token_buffer(self, tokens_tensor, tags_tensor):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self.batch_one_element_tensor(self.start_token_embedding, batch_size).unsqueeze(dim=0)
        start_tag_embedding = self.batch_one_element_tensor(self.start_tag_embedding, batch_size).unsqueeze(dim=0)
        start_word_embedding = self.token_tag2word(start_token_embedding, start_tag_embedding)
        # add tokens in reverse order
        word_embeddings = self.token_tag2embedding(tokens_tensor, tags_tensor).flip(dims=[0])
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        token_lengths = word_embeddings.size(0)
        token_lengths = torch.tensor([token_lengths], device=self.device, dtype=torch.long)
        state = self.token_buffer.inference_initialize(word_embeddings, token_lengths)
        return state

    def inference_update_token_buffer(self, state):
        """
        :type state: app.models.parallel_rnng.buffer_lstm.BufferState
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        batch_size = state.inputs.size(1)
        pop_op = self.pop_op(batch_size)
        state = self.token_buffer.inference_hold_or_pop(state, pop_op)
        return state

    def __str__(self):
        return (
            'DiscriminativeParallelRNNG(\n'
            + f'  action_history={self.action_history}\n'
            + f'  token_buffer={self.token_buffer}\n'
            + f'  stack={self.stack}\n'
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
