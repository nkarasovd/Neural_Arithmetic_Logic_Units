import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter

from typing import List

from models.model import GeneralModel


class NACCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W_hat = Parameter(Tensor(output_dim, input_dim))
        self.M_hat = Parameter(Tensor(output_dim, input_dim))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(x, W, self.bias)


class NAC(GeneralModel):
    def __init__(self, input_dim: int, output_dim: int,
                 num_hidden_layers: int, hidden_dim: int):
        super().__init__(input_dim, output_dim,
                         num_hidden_layers, hidden_dim)
        self.model = self.build_model()

    def layer(self, input_dim: int, output_dim: int) -> List[NACCell]:
        return [NACCell(input_dim, output_dim)]

    def name(self) -> str:
        return 'NAC'
