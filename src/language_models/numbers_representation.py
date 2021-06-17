from num2words import num2words

from typing import List, Union

NUMBER_CODES = {
    0: '',
    1: 'minus',
    2: 'zero',
    3: 'one',
    4: 'two',
    5: 'three',
    6: 'four',
    7: 'five',
    8: 'six',
    9: 'seven',
    10: 'eight',
    11: 'nine',
    12: 'ten',
    13: 'eleven',
    14: 'twelve',
    15: 'thirteen',
    16: 'fourteen',
    17: 'fifteen',
    18: 'sixteen',
    19: 'seventeen',
    20: 'eighteen',
    21: 'nineteen',
    22: 'twenty',
    23: 'thirty',
    24: 'forty',
    25: 'fifty',
    26: 'sixty',
    27: 'seventy',
    28: 'eighty',
    29: 'ninety',
    30: 'and',
    31: ',',
    32: 'hundred'
}


class Numbers:
    def __init__(self):
        self._EOS = 0
        self._number_codes = NUMBER_CODES
        self._reverse_codes = {v: k for k, v in self._number_codes.items()}
        self._language = 'en'

    def num2words(self, number: Union[int, float]) -> str:
        return num2words(int(number), lang=self._language)

    def _to_string(self, number: List[str]) -> str:
        return ' '.join(number).replace(' ,', ',')

    def to_string(self, numbers_lst: List[List[Union[int, float]]]) -> List[str]:
        # [[20, 0]] -> ['eighteen']
        return [self._to_string([self._number_codes[int(x)] for x in num_lst[:-1]]) for num_lst in numbers_lst]

    def _string2lst(self, x: str) -> List[str]:
        # twenty-five -> ['twenty', 'five']
        return x.replace('-', ' ').replace(',', ' ,').split(' ')

    def code_numbers(self, number_lst: List[Union[int, float]]) -> List[List[int]]:
        # [2, 3, 4, 18, 25] -> [[4, 0], [5, 0], [6, 0], [20, 0], [22, 7, 0]]
        return [[self._reverse_codes[w] for w in self._string2lst(self.num2words(num))] + [self._EOS]
                for num in number_lst]

    def __len__(self) -> int:
        return len(self._number_codes)
