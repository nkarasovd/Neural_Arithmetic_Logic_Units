import numpy as np

from torch import Tensor, LongTensor
from typing import Tuple, List
from language_models.numbers_representation import Numbers


class DataBuilder:
    def __init__(self):
        self._number_generator = Numbers()

    def generate_data(self, min_number: int = 0, max_number: int = 1000) -> Tuple[List[List[int]], List[int]]:
        numbers = [i for i in range(min_number, max_number)]
        numbers_representation = self._number_generator.code_numbers(numbers)
        return numbers_representation, numbers

    def split_data(self, data, size: Tuple = (169, 200, 631)):
        examples = ([*range(20), *range(20, 101, 10), 123], [], [])
        pass

    def shuffle_data(self, data: Tuple[List[List[int]], List[int]]) -> Tuple[Tensor, Tensor]:
        numbers_representation, numbers = data
        indices = np.random.permutation(len(numbers))

        x = [numbers_representation[i] for i in indices]
        y = [numbers[i] for i in indices]

        x = LongTensor(x).unsqueeze(1)
        y = LongTensor(y).unsqueeze(1)

        return x, y

    def to_string(self, number: List[int]) -> str:
        return self._number_generator.to_string([number])[0]

    def __len__(self) -> int:
        return len(self._number_generator)


if __name__ == '__main__':
    print(np.random.permutation(10))
