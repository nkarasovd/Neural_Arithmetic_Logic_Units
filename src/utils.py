import torch.nn as nn
from torch.nn import Hardtanh, Sigmoid, ReLU6, \
    Tanh, Tanhshrink, Hardshrink, LeakyReLU, Softshrink, Softsign, \
    Threshold, ReLU, PReLU, Softplus, ELU, SELU

from typing import Union, Type

activation_functions = Type[Union[Hardtanh, Sigmoid, ReLU6, Tanh,
                                  Tanhshrink, Hardshrink, LeakyReLU,
                                  Softshrink, Softsign, Threshold, ReLU,
                                  PReLU, Softplus, ELU, SELU]]

ACTIVATIONS = ['Hardtanh', 'Sigmoid', 'ReLU6', 'Tanh', 'Tanhshrink',
               'Hardshrink', 'LeakyReLU', 'Softshrink', 'Softsign',
               'Threshold', 'ReLU', 'PReLU', 'Softplus', 'ELU', 'SELU']


def string2function(name: str) -> activation_functions:
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
    else:
        raise ValueError('ERROR! Invalid function name!')
