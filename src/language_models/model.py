import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.init import xavier_uniform_

from typing import Tuple

from language_models.nalu_model import NALU


class LanguageModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int = 32,
                 lstm_layers: int = 1, nalu: bool = True, reduce_sum: bool = True):
        super().__init__()
        self.nalu = nalu
        self.reduce_sum = reduce_sum
        self.embedding_dim = embedding_dim
        self.lstm_layers = lstm_layers

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        if nalu:
            self.encoder = None
            self.final = NALU(embedding_dim * 2, 1, 1, 0)
        else:
            self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=lstm_layers)
            self.final = nn.Linear(embedding_dim, 1)

            with torch.no_grad():
                for w in self.encoder.parameters():
                    if w.dim() == 2:
                        xavier_uniform_(w)
                    else:
                        w.fill_(0)

    def nalu_forward(self, x: Tensor) -> Tensor:
        y = self.embedding(x)
        t = torch.zeros(1)

        for i in range(y.size(0) - 1):
            t += self.final(y[i: i + 2, 0].view(1, -1)).squeeze(1)

        return t

    def init_hidden(self, num_hidden: int, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(self.lstm_layers, batch_size, num_hidden),
                torch.zeros(self.lstm_layers, batch_size, num_hidden))

    def lstm_forward(self, x: Tensor) -> Tensor:
        hidden = self.init_hidden(self.embedding_dim)
        y = self.embedding(x)
        y, hidden = self.encoder(y, hidden)
        y = y.sum(dim=0) if self.reduce_sum else y[-1]
        y = self.final(y)[0]
        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.nalu:
            return self.nalu_forward(x)
        return self.lstm_forward(x)
