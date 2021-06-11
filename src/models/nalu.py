import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from typing import List

from models.nac import NACCell

from models.model import GeneralModel


# class NALUCell(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         self.eps = 1e-10
#
#         self.G = Parameter(torch.Tensor(self.output_dim, self.input_dim))
#         self.NACCell = NACCell(self.input_dim, self.output_dim)
#
#         self.register_parameter('G', self.G)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         a = self.NACCell(x)
#         g = torch.sigmoid(F.linear(x, self.G, bias=None))
#         m = torch.exp(self.NACCell(torch.log(torch.abs(x) + self.eps)))
#
#         return a * g + (1 - g) * m


class NALUCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.NACCell_1 = NACCell(self.input_dim, self.output_dim)
        self.NACCell_2 = NACCell(self.input_dim, self.output_dim)

        self.register_parameter('G', self.G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.NACCell_1(x)
        g = torch.sigmoid(F.linear(x, self.G, bias=None))
        m = torch.exp(self.NACCell_2(torch.log(torch.abs(x) + self.eps)))

        return a * g + (1 - g) * m


class NALU(GeneralModel):
    def __init__(self, input_dim: int, output_dim: int,
                 num_hidden_layers: int, hidden_dim: int):
        super().__init__(input_dim, output_dim,
                         num_hidden_layers, hidden_dim)
        self.model = self.build_model()

    def layer(self, input_dim: int, output_dim: int) -> List[NALUCell]:
        return [NALUCell(input_dim, output_dim)]

    def name(self) -> str:
        return 'NALU'
