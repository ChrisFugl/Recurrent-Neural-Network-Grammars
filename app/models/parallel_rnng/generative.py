from app.data.action_sets.generative import GenerativeActionSet
from app.models.parallel_rnng.parallel_rnng import ParallelRNNG
import torch
from torch import nn

class GenerativeParallelRNNG(ParallelRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size):
        """
        :type device: torch.device
        :type embeddings: app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.output_buffer_lstm.OutputBufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type sample_stack_size: int
        """
        action_set = GenerativeActionSet()
        generative = True
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size, action_set, generative)
        token_size = sizes[1]
        start_token_embedding = torch.FloatTensor(token_size).uniform_(-1, 1)
        self.start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, token_lengths):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: torch.Tensor
        """
        batch_size = tokens_tensor.size(1)
        start_token_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        push_op = self.push_op(batch_size)
        token_embeddings = self.token_embedding(tokens_tensor)
        token_embeddings = torch.cat((start_token_embedding, token_embeddings), dim=0)
        self.token_buffer.initialize(token_embeddings)
        self.token_buffer.hold_or_push(push_op)

    def get_word_embedding(self, state):
        """
        :type state: app.models.parallel_rnng.state.State
        :rtype: torch.Tensor, torch.Tensor
        """
        return self.token_embedding(state.token_index)

    def update_token_buffer(self, batch_size, token_action_indices, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type word_embeddings: torch.Tensor
        """
        op = self.hold_op(batch_size)
        op[token_action_indices] = 1
        self.token_buffer.hold_or_push(op)

    def __str__(self):
        return (
            'GenerativeParallelRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
