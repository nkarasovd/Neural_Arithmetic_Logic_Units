import torch
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from typing import List

from models.mlp import MLP

from utils import ACTIVATIONS, train, test

TRAIN_BORDERS = [-5, 6]
TEST_BORDERS = [-20, 21]
MAX_ITERATIONS = 10000
LEARNING_RATE = 1e-2


def get_mses(train_data: torch.Tensor, test_data: torch.Tensor) -> List[np.ndarray]:
    result = []

    for activation_func in ACTIVATIONS:
        print(f'Current activation: {activation_func}')
        mses = []
        for _ in range(100):
            cur_mlp = MLP(input_dim=1, output_dim=1,
                          num_hidden_layers=3, hidden_dim=8, activation=activation_func)
            optim = Adam(cur_mlp.parameters(), LEARNING_RATE)
            train(cur_mlp, optim, train_data, train_data, MAX_ITERATIONS)
            mses.append(test(cur_mlp, test_data, test_data))

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
    plt.savefig('./images/experiments/' + 'extrapolation_failure.png', format='png', dpi=300)
    plt.show()


def main():
    train_data = torch.arange(*TRAIN_BORDERS).unsqueeze_(1).float()
    test_data = torch.arange(*TEST_BORDERS).unsqueeze_(1).float()

    mses = get_mses(train_data, test_data)

    plot(mses)


if __name__ == '__main__':
    main()
