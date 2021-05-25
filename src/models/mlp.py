import torch.nn as nn

from typing import List, Union

from utils import string2function, activation_functions

from models.model import GeneralModel


class MLP(GeneralModel):
    def __init__(self, input_dim: int, output_dim: int,
                 num_hidden_layers: int, hidden_dim: int, activation: str = 'relu'):
        super().__init__(input_dim, output_dim,
                         num_hidden_layers, hidden_dim)
        self.activation = string2function(activation)
        self.model = self.build_model()

    def layer(self, input_dim: int, output_dim: int) -> \
            List[Union[nn.Linear, activation_functions]]:
        if output_dim != self.hidden_dim or self.activation is None:
            return [nn.Linear(input_dim, output_dim)]

        return [nn.Linear(input_dim, output_dim), self.activation()]
