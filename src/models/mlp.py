import torch
import torch.nn as nn

from typing import List, Union

from utils import string2function, activation_functions


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, output_dim: int,
                 hidden_dim: int, activation: str = 'relu'):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = string2function(activation)()

        self.model = self.build_model()

    def layer(self, input_dim: int, output_dim: int) -> \
            List[Union[nn.Linear, activation_functions]]:
        return [nn.Linear(input_dim, output_dim),
                self.activation]

    def build_model(self) -> nn.Sequential:
        layers = self.layer(self.input_dim, self.hidden_dim)

        for _ in range(self.num_layers - 1):
            layers += self.layer(self.hidden_dim, self.hidden_dim)

        layers += [nn.Linear(self.hidden_dim, self.output_dim)]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test_run():
    input_tensor = torch.tensor([0.1, 0.5, 0.2, 0.3])
    mlp = MLP(3, 4, 2, 3)
    output_tensor = mlp(input_tensor)
    return mlp.model, input_tensor, output_tensor


if __name__ == '__main__':
    model, input_tensor, output_tensor = test_run()
    print('Model:')
    print(model)
    print('Input tensor:')
    print(input_tensor)
    print('Output tensor:')
    print(output_tensor)
