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
        pass

    def shuffle_data(self, data):
        pass

    def to_string(self, number: List[int]) -> str:
        return self._number_generator.to_string([number])[0]
