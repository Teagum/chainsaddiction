"""
ChaisAddiction testing utilities
"""
from pathlib import Path
from typing import Union

import numpy as np


def gen_init_params(m_states: int, data: np.ndarray) -> tuple:
    """
    Generate initila parameters for HMM training.
    """
    init_lambda = gen_sdm(data, m_states)
    init_gamma = gen_prob_mat(m_states, m_states)
    init_delta = gen_prob_mat(1, m_states)
    return init_lambda, init_gamma, init_delta


def load_data(path: Union[str, Path], file: Union[str, Path],
              dtype: str = 'float128'):
    """
    Load test data from file.
    """
    path = Path(path)
    dfp = path.joinpath(file)
    return np.fromfile(str(dfp), dtype=dtype, sep='\n')


def gen_sdm(data: np.ndarray, m_states: int) -> np.ndarray:
    """
    Generate an initial guess of the state-dependend means.
    """
    return np.linspace(data.min(), data.max(), m_states)


def gen_prob_mat(rows: int, cols: int) -> np.ndarray:
    """
    Generate a random probability matrix.
    """
    var = np.exp(np.random.rand(rows, cols))
    return var / var.sum(axis=1, keepdims=True)
