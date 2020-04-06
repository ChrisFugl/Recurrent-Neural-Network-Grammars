from app.data.action_sets.discriminative import DiscriminativeActionSet
from app.models.rnng.rnng import RNNG
import torch
from torch import nn

class DiscriminativeRNNG(RNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads, pos_size, pos_embedding):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.stack.Stack, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type threads: int
        :type pos_size: int
        :type pos_embedding: torch.nn.Embedding
        """
        action_set = DiscriminativeActionSet()
        generative = False
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, threads, action_set, generative)
        self.pos_embedding = pos_embedding
        self.activation = nn.ReLU()
        token_size = sizes[1]
        rnn_input_size = sizes[2]
        self.word2buffer = nn.Linear(in_features=token_size + pos_size, out_features=rnn_input_size, bias=True)
        start_tag_embedding = torch.FloatTensor(1, 1, pos_size).uniform_(-1, 1)
        self.start_tag_embedding = nn.Parameter(start_tag_embedding, requires_grad=True)

    def shift(self, log_probs, outputs, action):
        word_embedding = outputs.token_top.data
        token_top = self.token_buffer.pop(outputs.token_top)
        stack_top = self.stack.push(word_embedding, data=action, top=outputs.stack_top)
        action_log_prob = self.get_base_log_prop(log_probs, self.shift_index)
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, token_top=token_top, token_counter=token_counter)

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, length):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.rnng.stack.StackNode
        """
        start_word_embedding = self.token_tag2word(self.start_token_embedding, self.start_tag_embedding)
        token_top = self.token_buffer.push(start_word_embedding, data=start_word_embedding)
        # discriminative model processes tokens in reverse order
        word_embeddings = self.get_word_embedding(tokens_tensor, tags_tensor)
        for index in range(length - 1, -1, -1):
            word_embedding = word_embeddings[index:index+1, :, :]
            token_top = self.token_buffer.push(word_embedding, data=word_embedding, top=token_top)
        return token_top

    def get_word_embedding(self, token, tag):
        token_embedding = self.token_embedding(token)
        tag_embedding = self.pos_embedding(tag)
        return self.token_tag2word(token_embedding, tag_embedding)

    def token_tag2word(self, token, tag):
        word = torch.cat((token, tag), dim=2)
        word = self.activation(self.word2buffer(word))
        return word

    def __str__(self):
        return (
            'DiscriminativeRNNG(\n'
            + f'  action_history={self.action_history}\n'
            + f'  token_buffer={self.token_buffer}\n'
            + f'  stack={self.stack}\n'
            + f'  representation={self.representation}\n'
            + f'  composer={self.composer}\n'
            + ')'
        )
