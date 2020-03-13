from app.constants import ACTION_GENERATE_INDEX, START_TOKEN_INDEX
from app.data.action_set.generative import Generative as GenerativeActionSet
from app.models.rnng.rnng import RNNG

class GenerativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, rnn_input_size, rnn_size, threads, token_distribution):
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
        :type token_distribution: app.distributions.distribution.Distribution
        """
        super().__init__(device, embeddings, structures, converters, representation, composer, rnn_input_size, rnn_size, threads)
        self._action_set = GenerativeActionSet()
        self._generative = True
        self._token_distribution = token_distribution

    def _generate(self, log_probs, outputs, action):
        token_index = self._token_converter.token2integer(action.argument)
        token_tensor = self._index2tensor(token_index)
        token_embedding = self._token_embedding(token_tensor)
        stack_top = self._stack.push(token_embedding, data=action, top=outputs.stack_top)
        token_top = self._token_buffer.push(token_embedding, top=outputs.token_top)
        if log_probs is None:
            action_log_prob = None
        else:
            generate_log_prob = self._get_base_log_prop(log_probs, ACTION_GENERATE_INDEX)
            token_log_prob = self._token_distribution.log_prob(log_probs.representation, action.argument)
            action_log_prob = generate_log_prob + token_log_prob
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
        start_token_embedding = self._token_embedding(start_token_tensor)
        token_top = self._token_buffer.push(start_token_embedding)
        return token_top

    def __str__(self):
        return (
            'GenerativeRNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + f'  token_distribution={self._token_distribution}\n'
            + ')'
        )
