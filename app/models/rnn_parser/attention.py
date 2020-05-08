from math import sqrt
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class Attention(nn.Module):

    def __init__(self, encoder_output_size, decoder_output_size):
        """
        :type encoder_output_size: int
        :type decoder_output_size: int
        """
        super().__init__()
        self.normalizer = sqrt(decoder_output_size)
        self.query = nn.Linear(in_features=decoder_output_size, out_features=encoder_output_size)
        self.key = nn.Linear(in_features=encoder_output_size, out_features=encoder_output_size)
        self.value = nn.Linear(in_features=encoder_output_size, out_features=encoder_output_size)
        self.embedding = nn.Linear(in_features=encoder_output_size, out_features=encoder_output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, encoder_lengths, decoder_state):
        max_length, batch_size, hidden_size = encoder_outputs.shape
        query = self.query(decoder_state).transpose(0, 1) # batch first
        key = self.key(encoder_outputs).transpose(0, 1) # batch first
        value = self.value(encoder_outputs).transpose(0, 1) # batch first
        # scaled dot attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.normalizer # batch_size, 1, max_length
        # mask out values beyond the desired lengths
        for i, length in enumerate(encoder_lengths):
            attention_scores[i, 0, length:] = MASKED_SCORE_VALUE
        attention_scores = attention_scores.view(batch_size, max_length)
        attention_weights = self.softmax(attention_scores)
        attention_weights_unsqueezed = attention_weights.view(batch_size, 1, max_length)
        value_weighted = torch.bmm(attention_weights_unsqueezed, value) # batch_size, 1, hidden_size
        embedding = self.embedding(value_weighted)
        output = embedding.transpose(0, 1) # length first
        return output, attention_weights
