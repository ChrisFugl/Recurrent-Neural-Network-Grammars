from app.dropout.variational import variational_dropout_mask
from app.utils import batched_index_select
from math import sqrt
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class Attention(nn.Module):

    def __init__(self, device, embedding_size, dropout, dropout_type):
        """
        :type device: torch.device
        :type embedding_size: int
        """
        super().__init__()
        self.device = device
        self.normalizer = sqrt(embedding_size)
        self.query = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.key = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.value = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.embedding = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.dropout = dropout
        dropout_p = 0.0 if dropout is None else dropout
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.use_normal_dropout = dropout is not None and dropout_type == 'normal'
        self.use_variational_dropout = dropout is not None and dropout_type == 'variational'
        self.embedding_size = embedding_size

    def forward(self, inputs, lengths, query=None):
        """
        :type inputs: torch.Tensor
        :type lengths: torch.Tensor
        :type query: torch.Tensor
        :rtype: torch.Tensor, torch.Tensor
        """
        max_length, batch_size, hidden_size = inputs.shape
        if query is None:
            query_inputs = batched_index_select(inputs, lengths - 1) # 1, batch_size, hidden_size
        else:
            query_inputs = query
        query = self.query(query_inputs).transpose(0, 1) # batch first
        key = self.key(inputs).transpose(0, 1) # batch first
        value = self.value(inputs).transpose(0, 1) # batch first
        # scaled dot attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.normalizer # batch_size, 1, max_length
        # mask out values beyond the desired lengths
        for i, length in enumerate(lengths):
            attention_scores[i, 0, length:] = MASKED_SCORE_VALUE
        attention_scores = attention_scores.view(batch_size, max_length)
        attention_weights = self.softmax(attention_scores)
        attention_weights_unsqueezed = attention_weights.view(batch_size, 1, max_length)
        value_weighted = torch.bmm(attention_weights_unsqueezed, value) # batch_size, 1, hidden_size
        embedding = self.embedding(value_weighted)
        output = embedding.transpose(0, 1) # length first
        output = self.activation(output)
        if self.use_normal_dropout:
            output = self.dropout_layer(output)
        elif self.use_variational_dropout and self.training:
            output = output * self.dropout_mask.expand(1, batch_size, -1)
        return output, attention_weights

    def reset(self):
        if self.use_variational_dropout and self.training:
            self.dropout_mask = variational_dropout_mask((1, 1, self.embedding_size), self.dropout, device=self.device)
