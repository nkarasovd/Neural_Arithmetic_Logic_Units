import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from torch import Tensor
from torch.nn.parameter import Parameter

from typing import List

from models.model import GeneralModel


class NACCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W_hat = Parameter(Tensor(output_dim, input_dim))
        self.M_hat = Parameter(Tensor(output_dim, input_dim))
        self.W = Parameter(F.tanh(self.W_hat) * F.sigmoid(self.M_hat))

        # self.register_parameter('W_hat', self.W_hat)
        # self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        # W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(x, self.W, self.bias)


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
