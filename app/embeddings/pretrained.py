from app.constants import PAD_INDEX
from app.data.batch_utils import sequences2tensor
from app.dropout.embedding import embedding_dropout_mask
from app.embeddings.embedding import Embedding
import hydra
import torch
from torch import nn

class PretrainedEmbedding(Embedding):

    def __init__(self, device, path, dropout):
        """
        :type device: torch.device
        :type path: str
        :type dropout: float
        """
        super().__init__()
        self.device = device
        self.path = path
        self.dropout = dropout
        self.weights, self.num_embeddings, self.embedding_size, self.token2index = self.load_from_path(device, path)
        self.use_dropout = dropout is not None
        self.reset()

    def forward(self, tokens):
        """
        :type tokens: list of list of str
        :rtype: torch.Tensor
        """
        tensor = sequences2tensor(self.device, self.token2int, tokens)
        return nn.functional.embedding(tensor, self.masked_weights, padding_idx=PAD_INDEX)

    def reset(self):
        if self.use_dropout and self.training:
            mask = embedding_dropout_mask(self.weights, self.dropout)
            self.masked_weights = mask * self.weights
        else:
            self.masked_weights = self.weights

    def size(self):
        """
        :rtype: int
        """
        return self.embedding_size

    def load_from_path(self, device, path):
        absolute_path = hydra.utils.to_absolute_path(path)
        with open(absolute_path, 'r') as file:
            lines = file.read().strip().split('\n')
        num_embeddings, embedding_size = tuple(map(int, lines[0].strip().split(' ')))
        num_embeddings += 1
        weights = torch.zeros((num_embeddings, embedding_size), device=device, dtype=torch.float, requires_grad=False)
        token2index = {}
        for index, line in enumerate(lines[1:], start=1):
            split = line.strip().split(' ')
            token = split[0]
            vector = list(map(float, split[1:]))
            weights[index] = torch.tensor(vector, device=device, dtype=torch.float)
            token2index[token] = index
        return weights, num_embeddings, embedding_size, token2index

    def token2int(self, token):
        try:
            return self.token2index[token]
        except KeyError:
            return PAD_INDEX

    def __str__(self):
        if self.use_dropout:
            return f'Pretrained(path={self.path}, dropout={self.dropout})'
        else:
            return f'Pretrained(path={self.path})'
