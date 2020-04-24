from app.data.action_sets.generative import GenerativeActionSet
from app.models.rnng.rnng import RNNG
import torch

class GenerativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads, token_distribution):
        """
        :type device: torch.device
        :type embeddings: app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.buffer.Buffer, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type threads: int
        :type token_distribution: app.distributions.distribution.Distribution
        """
        action_set = GenerativeActionSet()
        generative = True
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, threads, action_set, generative)
        self.token_distribution = token_distribution

    def generate(self, log_probs, tokens, tags, outputs, action):
        stack_top = outputs.stack_top
        buffer_state = outputs.buffer_state
        if self.uses_stack or self.uses_buffer:
            if self.uses_buffer:
                buffer_state = self.token_buffer.push(buffer_state)
            if self.uses_stack:
                token_index = self.token_converter.token2integer(action.argument)
                token_tensor = torch.tensor([[token_index]], device=self.device, dtype=torch.long)
                token_embedding = self.token_embedding(token_tensor)
                stack_top = self.stack.push(token_embedding, data=action, top=stack_top)
        if log_probs is None:
            action_log_prob = None
        else:
            generate_log_prob = self.get_base_log_prop(log_probs, self.gen_index)
            token_log_prob = self.token_distribution.log_prob(log_probs.representation, action.argument)
            action_log_prob = generate_log_prob + token_log_prob
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, buffer_state=buffer_state, token_counter=token_counter)

    def initialize_token_buffer(self, tokens, tags, lengths):
        """
        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of app.models.rnng.buffer.BufferState
        """
        batch_size = lengths.size(0)
        start_word_embedding = self.start_token_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        word_embeddings = self.token_embedding(tokens)
        word_embeddings = torch.cat((start_word_embedding, word_embeddings), dim=0)
        start_indices = [0 for _ in range(batch_size)]
        # plus one to account for start embedding
        buffer_states = self.token_buffer.initialize(word_embeddings, lengths + 1, start_indices)
        return buffer_states

    def __str__(self):
        return (
            'GenerativeRNNG(\n'
            + ('' if not self.uses_history else f'  action_history={self.action_history}\n')
            + ('' if not self.uses_buffer else f'  token_buffer={self.token_buffer}\n')
            + ('' if not self.uses_stack else f'  stack={self.stack}\n')
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + f'  token_distribution={self.token_distribution}\n'
            + ')'
        )
