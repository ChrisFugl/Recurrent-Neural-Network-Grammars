from app.representations.representation import Representation
from app.utils import batched_index_select
from math import sqrt
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class AttentiveRepresentation(Representation):

    def __init__(self, embedding_size, representation_size, dropout):
        """
        :type embedding_size: int
        :type representation_size: int
        :type dropout: float
        """
        super().__init__()
        self.representation_size = representation_size
        self.normalizer = sqrt(embedding_size)
        self.his2query = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.his2key = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.his2value = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.his2embedding = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.stack2query = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.stack2key = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.stack2value = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.stack2embedding = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.buf2query = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.buf2key = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.buf2value = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.buf2embedding = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.softmax = nn.Softmax(dim=1)
        # token, stack, action
        input_size = 3 * embedding_size
        self.embedding2representation = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, his, his_lengths, stack, stack_lengths, buf, buf_lengths):
        """
        :type his: torch.Tensor
        :type his_lengths: torch.Tensor
        :type stack: torch.Tensor
        :type stack_lengths: torch.Tensor
        :type buf: torch.Tensor
        :type buf_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        his_embedding = self.embed(self.his2query, self.his2key, self.his2value, self.his2embedding, his, his_lengths)
        stack_embedding = self.embed(self.stack2query, self.stack2key, self.stack2value, self.stack2embedding, stack, stack_lengths)
        buf_embedding = self.embed(self.buf2query, self.buf2key, self.buf2value, self.buf2embedding, buf, buf_lengths)
        embeddings = [his_embedding, stack_embedding, buf_embedding]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        output = torch.cat(embeddings, dim=2)
        output = self.embedding2representation(output)
        output = self.activation(output)
        return output

    def top_only(self):
        """
        :rtype: bool
        """
        return False

    def uses_action_history(self):
        """
        :rtype: bool
        """
        return True

    def uses_stack(self):
        """
        :rtype: bool
        """
        return True

    def uses_token_buffer(self):
        """
        :rtype: bool
        """
        return True

    def embed(self, inputs2query, inputs2key, inputs2value, value2embedding, inputs, lengths):
        max_length, batch_size, hidden_size = inputs.shape
        query_inputs = batched_index_select(inputs, lengths - 1) # 1, batch_size, hidden_size
        query = inputs2query(query_inputs).transpose(0, 1) # batch first
        key = inputs2key(inputs).transpose(0, 1) # batch first
        value = inputs2value(inputs).transpose(0, 1) # batch first
        # scaled dot attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.normalizer # batch_size, 1, max_length
        # mask out values beyond the desired lengths
        for i, length in enumerate(lengths):
            attention_scores[i, 0, length:] = MASKED_SCORE_VALUE
        attention_scores = attention_scores.view(batch_size, max_length)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, 1, max_length)
        value_weighted = torch.bmm(attention_weights, value) # batch_size, 1, hidden_size
        embedding = value2embedding(value_weighted)
        embedding = embedding.transpose(0, 1) # length first
        output = self.activation(embedding)
        output = self.dropout(output)
        return output

    def __str__(self):
        return f'Attentive(size={self.representation_size})'
