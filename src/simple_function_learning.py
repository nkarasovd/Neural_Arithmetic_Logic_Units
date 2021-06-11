import numpy as np

ARITHMETIC_FUNCTIONS = {
    'a + b': lambda a, b: a + b,
    'a - b': lambda a, b: a - b,
    'a * b': lambda a, b: a * b,
    'a / b': lambda a, b: a / b,
    'a ^ 2': lambda a, _: a ** 2,
    'sqrt(a)': lambda a, _: np.aqrt(a),
}
