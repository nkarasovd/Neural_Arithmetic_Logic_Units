import torch.nn as nn

from models.nalu import NALUCell


class NALU:
    def __init__(self, input_dim: int, output_dim: int,
                 num_layers: int, hidden_dim: int):
        layers = [NALUCell(hidden_dim if n > 0 else input_dim,
                           hidden_dim if n < num_layers - 1 else output_dim) for n in range(num_layers)]

        self.model = nn.Sequential(*layers)

    def name(self) -> str:
        return 'NALU_language'
