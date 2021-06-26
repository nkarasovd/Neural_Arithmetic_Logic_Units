import torch.nn as nn

from torch import Tensor

from models.nalu import NALUCell


class NALU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 num_layers: int, hidden_dim: int):
        super().__init__()
        layers = [NALUCell(hidden_dim if n > 0 else input_dim,
                           hidden_dim if n < num_layers - 1 else output_dim) for n in range(num_layers)]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def name(self) -> str:
        return 'NALU_language'
