import numpy as np

import torch
import torch.nn.functional as F

from torch import Tensor, FloatTensor

from typing import List, Tuple

from models.mlp import MLP
from models.nac import NAC
from models.nalu import NALU

from utils import train

DATA = Tuple[np.ndarray, np.ndarray, np.ndarray,
             np.ndarray, np.ndarray, np.ndarray, float, float]

ARITHMETIC_FUNCTIONS = {
    'a + b': lambda a, b: a + b,
    'a - b': lambda a, b: a - b,
    'a * b': lambda a, b: a * b,
    'a / b': lambda a, b: a / b,
    'a ^ 2': lambda a, _: a ** 2,
    'sqrt(a)': lambda a, _: np.sqrt(a),
}

VECTOR_SIZE = 100
M, N = 0, 10
P, Q = 60, 100


def generate_a_b_y(operator: str, right_bound: float, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.random.uniform(0, right_bound, size=(size, VECTOR_SIZE))
    a, b = np.sum(x[:, M:N], axis=1), np.sum(x[:, P:Q], axis=1)
    y = ARITHMETIC_FUNCTIONS[operator](a, b)
    return a, b, y


def generate_test_data(operator: str, right_bound: float,
                       test_size: int, max_values: List[float],
                       interpolation: bool) -> Tuple[np.ndarray, np.ndarray]:
    a_max, b_max, y_max = max_values
    x_test, y_test = [], []

    test_len = 0

    while test_len < test_size:
        a, b, y = generate_a_b_y(operator, right_bound, test_size)

        in_range = (a < a_max) & (b < b_max) & (y < y_max)

        if interpolation:
            indices = in_range
        else:
            indices = ~in_range

        x_test.append(np.array([(a_, b_) for a_, b_ in zip(a, b)])[indices])
        y_test.append(y[indices])

        test_len += np.sum(indices)

    x_test = np.concatenate(x_test)[:test_size].astype(np.float32)
    y_test = np.concatenate(y_test)[:test_size].astype(np.float32)

    return x_test, y_test


def random_baseline(x_test_i: np.ndarray, y_test_i: np.ndarray,
                    x_test_e: np.ndarray, y_test_e: np.ndarray,
                    n_repeat: int = 20) -> Tuple[float, float]:
    total_rmse_i, total_rmse_e = 0, 0

    for i in range(n_repeat):
        net = MLP(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2, activation='relu')
        x = FloatTensor(x_test_i)
        with torch.no_grad():
            outs = net(x)
            rmse_i = torch.sqrt(F.mse_loss(outs, FloatTensor(y_test_i).view(-1, 1)))
            outs = net(FloatTensor(x_test_e))
            rmse_e = torch.sqrt(F.mse_loss(outs, FloatTensor(y_test_e).view(-1, 1)))

            total_rmse_i += rmse_i
            total_rmse_e += rmse_e

    return total_rmse_i / n_repeat, total_rmse_e / n_repeat


def generate_data(operator: str, right_bound: float,
                  train_size: int, test_size: int) -> DATA:
    a, b, y_train = generate_a_b_y(operator, right_bound, train_size)

    a_max, b_max = np.max(a), np.max(b)
    y_max = np.max(y_train)

    x_test_i, y_test_i = generate_test_data(operator, right_bound, test_size, [a_max, b_max, y_max], True)
    x_test_e, y_test_e = generate_test_data(operator, right_bound * 5, test_size, [a_max, b_max, y_max], False)

    random_rmse_i, random_rmse_e = random_baseline(x_test_i, y_test_i, x_test_e, y_test_e)
    x_train = np.array([(a_, b_) for a_, b_ in zip(a, b)])
    return x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e, random_rmse_i, random_rmse_e


def test(model, data: Tensor, target: Tensor):
    with torch.no_grad():
        outs = model(data)
        return torch.sqrt(F.mse_loss(outs, target.view(-1, 1)))


def my_main():
    models = [
        # MLP(input_dim=2,
        #     output_dim=1,
        #     num_hidden_layers=1,
        #     hidden_dim=2,
        #     activation='relu6'),
        # MLP(input_dim=2,
        #     output_dim=1,
        #     num_hidden_layers=1,
        #     hidden_dim=2,
        #     activation='none'),
        NAC(input_dim=2,
            output_dim=1,
            num_hidden_layers=1,
            hidden_dim=2),
        NALU(input_dim=2,
             output_dim=1,
             num_hidden_layers=1,
             hidden_dim=2)
    ]

    results_i, results_e = {}, {}

    for fn_str in ARITHMETIC_FUNCTIONS.keys():
        print(f'[!] Testing function: {fn_str}')

        results_i[fn_str], results_e[fn_str] = [], []

        x_train, y_train, x_test_i, y_test_i, \
        x_test_e, y_test_e, r_i, r_e = generate_data(fn_str, 0.5, 50000, 5000)

        for net in models:
            print(f"\t> Training {str(net)}")
            optim = torch.optim.Adam(net.parameters(), lr=1e-2)
            train(net, optim, FloatTensor(x_train), FloatTensor(y_train), int(1e5))
            mse = test(net, FloatTensor(x_test_i), FloatTensor(y_test_i)).item()
            results_i[fn_str].append(mse / r_i * 100)

    with open('./' + "interpolation.txt", "w") as f:
        f.write("Relu6\tNone\tNAC\tNALU\n")
        for k, v in results_i.items():
            f.write("{:.3f}\t{:.3f}\t\n".format(*results_i[k]))


if __name__ == '__main__':
    my_main()
    # models = [
    #     MLP(input_dim=2,
    #         output_dim=1,
    #         num_hidden_layers=1,
    #         hidden_dim=2,
    #         activation='relu6'),
    #     MLP(input_dim=2,
    #         output_dim=1,
    #         num_hidden_layers=1,
    #         hidden_dim=2,
    #         activation='none'),
    #     NAC(input_dim=2,
    #         output_dim=1,
    #         num_hidden_layers=1,
    #         hidden_dim=2),
    #     NALU(input_dim=2,
    #          output_dim=1,
    #          num_hidden_layers=1,
    #          hidden_dim=2)
    # ]
    #
    # for m in models:
    #     for c in m.parameters():
    #         print(c)
    #     print()