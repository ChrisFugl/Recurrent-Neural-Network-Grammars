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
        start_token_embedding = torch.FloatTensor(1, 1, token_size).uniform_(-1, 1)
        self._start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)

    def _initialize_token_buffer(self, tokens_tensor, tags_tensor):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        batch_size = tokens_tensor.size(1)
        push_all_op = self._push_op(batch_size)
        start_token_embedding = self._batch_one_element_tensor(self._start_token_embedding, batch_size)
        token_buffer = self._token_buffer.initialize(batch_size)
        token_buffer = self._token_buffer(token_buffer, start_token_embedding, push_all_op)
        return token_buffer

    def __str__(self):
        return (
            'GenerativeParallelRNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + ')'
        )
