import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Hardtanh, Sigmoid, ReLU6, \
    Tanh, Tanhshrink, Hardshrink, LeakyReLU, Softshrink, Softsign, \
    Threshold, ReLU, PReLU, Softplus, ELU, SELU

from typing import Union, Type, Optional

from models.model import GeneralModel

activation_functions = Type[Union[Hardtanh, Sigmoid, ReLU6, Tanh,
                                  Tanhshrink, Hardshrink, LeakyReLU,
                                  Softshrink, Softsign, Threshold, ReLU,
                                  PReLU, Softplus, ELU, SELU]]

ACTIVATIONS = ['Hardtanh', 'Sigmoid', 'ReLU6', 'Tanh', 'Tanhshrink',
               'Hardshrink', 'LeakyReLU', 'Softshrink', 'Softsign',
               'ReLU', 'PReLU', 'Softplus', 'ELU', 'SELU']  # 'Threshold'


def string2function(name: str) -> Optional[activation_functions]:
    name = name.lower()
    if name == 'hardtanh':
        return nn.Hardtanh
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'relu6':
        return nn.ReLU6
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'tanhshrink':
        return nn.Tanhshrink
    elif name == 'hardshrink':
        return nn.Hardshrink
    elif name == 'leakyrelu':
        return nn.LeakyReLU
    elif name == 'softshrink':
        return nn.Softshrink
    elif name == 'softsign':
        return nn.Softsign
    elif name == 'threshold':
        return nn.Threshold
    elif name == 'relu':
        return nn.ReLU
    elif name == 'prelu':
        return nn.PReLU
    elif name == 'softplus':
        return nn.Softplus
    elif name == 'elu':
        return nn.ELU
    elif name == 'selu':
        return nn.SELU
    elif name == 'none':
        return None
    else:
        raise ValueError('ERROR! Invalid function name!')


def train(model: GeneralModel, optimizer: torch.optim.Optimizer, data: torch.Tensor,
          target: torch.Tensor, iterations: int, verbose: bool = False):
    for iter in range(iterations):
        optimizer.zero_grad()

        outs = model(data)

        loss = F.mse_loss(outs, target)

        loss.backward()
        optimizer.step()

        if verbose and (iter + 1) % 1000 == 0:
            MAE = torch.mean(torch.abs(outs - data))
            print(f'Iter: {iter + 1}, Loss: {loss}, MAE: {MAE}')


def test(model: GeneralModel, data: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        outs = model(data)
        return torch.abs(outs - target)
