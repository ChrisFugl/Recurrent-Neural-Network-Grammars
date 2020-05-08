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
        self.encoder_output_size = encoder_output_size
        self.decoder_output_size = decoder_output_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.encoder_affine = nn.Linear(in_features=encoder_output_size, out_features=encoder_output_size)
        self.decoder_affine = nn.Linear(in_features=decoder_output_size, out_features=encoder_output_size)
        scale_vector = torch.FloatTensor(encoder_output_size).normal_()
        self.scale_vector = nn.Parameter(scale_vector, requires_grad=True)

    def forward(self, encoder_outputs, encoder_lengths, decoder_state):
        """
        :param encoder_outputs: sentence length x batch size x encoder output size
        :type encoder_outputs: torch.Tensor
        :param decoder_state: 1 x batch size x decoder output size
        :type encoder_lengths: torch.Tensor
        :type decoder_state: torch.Tensor
        :rtype: torch.Tensor, torch.Tensor
        """
        sentence_length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        encoder_outputs_batch_first = encoder_outputs.transpose(0, 1)
        decoder_state_batch_first = decoder_state.transpose(0, 1)
        encoded_transformed = self.encoder_affine(encoder_outputs_batch_first)
        decoded_transformed = self.decoder_affine(decoder_state_batch_first).expand(batch_size, sentence_length, -1)
        tanh_input = encoded_transformed + decoded_transformed
        tanh_output = self.tanh(tanh_input) # batch size, sentence length, encoder output size
        scale_matrix = self.scale_vector.view(1, -1, 1).expand(batch_size, -1, 1)
        softmax_input = torch.bmm(tanh_output, scale_matrix).transpose(1, 2) # batch size, 1, sentence length
        for i, length in enumerate(encoder_lengths):
            softmax_input[i, 0, length:] = MASKED_SCORE_VALUE
        weights = self.softmax(softmax_input)
        output = torch.bmm(weights, encoder_outputs_batch_first).transpose(0, 1) # 1, batch size, encoder output size
        return output, weights
