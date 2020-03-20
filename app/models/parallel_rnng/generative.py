from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
import torch
from torch import nn

class GenerativeParallelRNNG(ParallelRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.stack_lstm.StackLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        """
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes)
        token_size = sizes[1]
        start_token_embedding = torch.FloatTensor(1, token_size).uniform_(-1, 1)
        self.start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, token_lengths):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self.batch_one_element_tensor(self.start_token_embedding, batch_size).unsqueeze(dim=0)
        token_embeddings = self.token_embedding(tokens_tensor)
        token_embeddings = torch.cat((start_token_embedding, token_embeddings), dim=0)
        indices = torch.tensor([0] * batch_size, device=self.device, dtype=torch.long)
        token_buffer = self.token_buffer.initialize(token_embeddings, indices)
        return token_buffer

    def get_word_embedding(self, preprocessed, token_action_indices):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type token_action_indices: torch.Tensor
        :rtype: torch.Tensor, torch.Tensor
        """
        token_indices = preprocessed.token_index[token_action_indices]
        word_embeddings = self.token_embedding(token_indices)
        return word_embeddings

    def update_token_buffer(self, batch_size, token_action_indices, token_buffer, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type token_buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :type word_embeddings: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        op = self.hold_op(batch_size)
        op[token_action_indices] = 1
        token_buffer = self.token_buffer.hold_or_push(token_buffer, op)
        return token_buffer

    def __str__(self):
        return (
            'GenerativeParallelRNNG(\n'
            + f'  action_history={self.action_history}\n'
            + f'  token_buffer={self.token_buffer}\n'
            + f'  stack={self.stack}\n'
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
