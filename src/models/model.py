import torch
import torch.nn as nn

from typing import List


class GeneralModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 num_hidden_layers: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self.model = None

    def layer(self, input_dim: int, output_dim: int) -> List:
        raise NotImplementedError

    def build_model(self) -> nn.Sequential:
        layers = self.layer(self.input_dim, self.hidden_dim)

        for _ in range(self.num_hidden_layers):
            layers += self.layer(self.hidden_dim, self.hidden_dim)

        layers += self.layer(self.hidden_dim, self.output_dim)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
