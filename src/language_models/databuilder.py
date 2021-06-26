import numpy as np

import torch
from torch import Tensor, FloatTensor, LongTensor

from random import shuffle

from typing import Tuple, List, Union

from language_models.numbers_representation import Numbers


class DataBuilder:
    def __init__(self):
        self._number_generator = Numbers()

    def generate_data(self, min_number: int = 0, max_number: int = 1000) -> Tuple[List[List[int]], Tensor]:
        numbers = [i for i in range(min_number, max_number)]
        numbers_representation = self._number_generator.code_numbers(numbers)
        return numbers_representation, FloatTensor(numbers)

    def split_data(self, data: Tuple[List[List[int]], Tensor],
                   size: Tuple = (169, 200, 631)) -> List[List[Union[List[Tensor], Tensor]]]:
        include = ([*range(20), *range(20, 101, 10), 123], [], [])
        x, y = data

        result = []

        for inc in include:
            x_ = [LongTensor(x[j]).unsqueeze(1) for j in inc]
            y_ = [y[j] for j in inc]
            result.append([x_, y_])

        indices = np.setdiff1d(list(range(sum(size))), include[0])

        shuffle(indices)

        size_before = 0

        for i, s in enumerate(size):
            s -= len(include[i])
            x_ = [LongTensor(x[indices[idx]]).unsqueeze(1) for idx in range(size_before, s + size_before)]
            y_ = [y[indices[idx]] for idx in range(size_before, s + size_before)]
            result[i] = [r + a for r, a in zip(result[i], [x_, y_])]
            size_before += s

        result = [[r[0], torch.Tensor(r[1])] for r in result]

        return result

    def shuffle_data(self, data: Tuple[List[Tensor], Tensor]) \
            -> Tuple[List[Tensor], Tensor]:
        numbers_representation, numbers = data
        rand = torch.randperm(numbers.size(0))
        numbers_representation = [numbers_representation[r] for r in rand]
        numbers = numbers[rand]

        return numbers_representation, numbers

    def to_string(self, number: List[int]) -> str:
        return self._number_generator.to_string([number])[0]

    def __len__(self) -> int:
        return len(self._number_generator)
