from app.data.action_sets.generative import GenerativeActionSet
from app.models.rnng.rnng import RNNG

class GenerativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads, token_distribution):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.stack.Stack, app.models.rnng.stack.Stack
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

    def generate(self, log_probs, outputs, action):
        token_index = self.token_converter.token2integer(action.argument)
        token_tensor = self.index2tensor(token_index)
        token_embedding = self.token_embedding(token_tensor)
        stack_top = self.stack.push(token_embedding, data=action, top=outputs.stack_top)
        token_top = self.token_buffer.push(token_embedding, top=outputs.token_top)
        if log_probs is None:
            action_log_prob = None
        else:
            generate_log_prob = self.get_base_log_prop(log_probs, self.gen_index)
            token_log_prob = self.token_distribution.log_prob(log_probs.representation, action.argument)
            action_log_prob = generate_log_prob + token_log_prob
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, token_top=token_top, token_counter=token_counter)

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, length):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.rnng.stack.StackNode
        """
        token_top = self.token_buffer.push(self.start_token_embedding.view(1, 1, -1))
        return token_top

    def __str__(self):
        return (
            'GenerativeRNNG(\n'
            + f'  action_history={self.action_history}\n'
            + f'  token_buffer={self.token_buffer}\n'
            + f'  stack={self.stack}\n'
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + f'  token_distribution={self.token_distribution}\n'
            + ')'
        )
