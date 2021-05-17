import torch
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from typing import List

from models.mlp import MLP

from utils import ACTIVATIONS

TRAIN_BORDERS = [-5, 6]
TEST_BORDERS = [-20, 21]
MAX_ITERATIONS = 10000


def train(model: MLP, optimizer: torch.optim.Optimizer, data: torch.Tensor,
          iterations: int = MAX_ITERATIONS, verbose: bool = False):
    for iter in range(iterations):
        optimizer.zero_grad()

        outs = model(data)

        loss = F.mse_loss(outs, data)

        loss.backward()
        optimizer.step()

        if verbose and (iter + 1) % 1000 == 0:
            MAE = torch.mean(torch.abs(outs - data))
            print(f'Iter: {iter + 1}, Loss: {loss}, MAE: {MAE}')


def test(model: MLP, data: torch.Tensor):
    with torch.no_grad():
        outs = model(data)
        return torch.abs(outs - data)


def get_mses(train_data: torch.Tensor, test_data: torch.Tensor) -> List[np.ndarray]:
    result = []

    for activation_func in ACTIVATIONS:
        mses = []
        for _ in range(1):
            cur_mlp = MLP(input_dim=1, output_dim=1,
                          num_hidden_layers=3, hidden_dim=8, activation=activation_func)
            optim = Adam(cur_mlp.parameters(), 1e-2)
            train(cur_mlp, optim, train_data)
            mses.append(test(cur_mlp, test_data))

        result.append(torch.cat(mses, dim=1).mean(dim=1))

    result = [x.numpy().flatten() for x in result]

    return result


def plot(mses: List[np.ndarray]):
    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.arange(*TEST_BORDERS)

    for mse, activation in zip(mses, ACTIVATIONS):
        ax.plot(x, mse, label=activation)

    plt.grid()
    plt.legend(loc='best')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('./images/' + 'extrapolation.png', format='png', dpi=300)
    plt.show()


def main():
    train_data = torch.arange(*TRAIN_BORDERS).unsqueeze_(1).float()
    test_data = torch.arange(*TEST_BORDERS).unsqueeze_(1).float()

    mses = get_mses(train_data, test_data)

    plot(mses)


if __name__ == '__main__':
    main()
