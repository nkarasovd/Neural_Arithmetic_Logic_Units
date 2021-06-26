import numpy as np

import torch
from torch import Tensor, LongTensor, FloatTensor
from typing import Tuple, List
from random import shuffle

from language_models.numbers_representation import Numbers


class DataBuilder:
    def __init__(self):
        self._number_generator = Numbers()

    def generate_data(self, min_number: int = 0, max_number: int = 1000) -> Tuple[List[List[int]], Tensor]:
        numbers = [i for i in range(min_number, max_number)]
        numbers_representation = self._number_generator.code_numbers(numbers)
        return numbers_representation, FloatTensor(numbers)
        # y = torch.arange(min_number, max_number, dtype=torch.float)
        # x = self._number_generator.code_numbers(y)
        # return x, y

    def split_data(self, data, size: Tuple = (169, 200, 631)):
        include = ([*range(20), *range(20, 101, 10), 123], [], [])
        # x, y = data
        #
        # res = []
        #
        # res += [[np.array(x)[examples[0]].tolist(), np.array(y)[examples[0]].tolist()]]
        #
        # res += [[[], []], [[], []]]
        #
        # indices = list(range(1000))
        #
        # ids = np.setdiff1d(indices, examples[0])
        #
        # shuffle(ids)
        #
        # size_before = 0
        #
        # for i, s in enumerate(size):
        #     s -= len(examples[i])
        #     x_lst, y_lst = [], []
        #     for j in range(size_before, s + size_before):
        #         x_lst.append(x[ids[j]])
        #         y_lst.append(y[ids[j]])
        #
        #     res[i] = [r + a for r, a in zip(res[i], [x_lst, y_lst])]
        #     size_before += s
        #
        # res = [[r[0], Tensor(r[1])] for r in res]
        #
        # return res
        result = [[[d[j] for j in inc] for d in data] for inc in include]
        indices = list(range(sum(size)))
        for idx in sorted([j for sub in include for j in sub], reverse=True):
            del indices[idx]
        shuffle(indices)
        size_before = 0
        for i, s in enumerate(size):
            s -= len(include[i])
            append = [[d[indices[idx]] for idx in range(size_before, s + size_before)] for d in data]
            result[i] = [r + a for r, a in zip(result[i], append)]
            size_before += s
        result = [[r[0], torch.Tensor(r[1])] for r in result]
        return result

    def shuffle_data(self, data: Tuple[List[List[int]], List[int]]):
        # numbers_representation, numbers = data
        # indices = np.random.permutation(len(numbers))
        #
        # x = [LongTensor(numbers_representation[i]).unsqueeze(1) for i in indices]
        #
        # y = [numbers[i] for i in indices]
        #
        # return x, y

        data_x, data_y = data
        rand = torch.randperm(data_y.size(0))
        data_x = [data_x[r] for r in rand]
        data_y = data_y[rand]
        return data_x, data_y

    def to_string(self, number: List[int]) -> str:
        return self._number_generator.to_string([number])[0]

    def __len__(self) -> int:
        return len(self._number_generator)


if __name__ == '__main__':
    c = torch.arange(0, 10, dtype=torch.float)
    print(c)
    print(type(c))
    print(c.size())

    print()

    d = [i for i in range(0, 10)]
    d_t = torch.FloatTensor(d)
    print(d_t)
    print(type(d_t))
    print(d_t.size())
    print()
    dat = DataBuilder()
    x, y = dat.generate_data(max_number=10)
    print(type(x))
    print(y)

    for i in y:
        print(i.unsqueeze(0), i.unsqueeze(0).size())
