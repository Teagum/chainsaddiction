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
    init_lambda = utils.StateDependentMeansInitializer.hist(data, m_states)
    init_gamma = utils.TpmInitializer.softmax(m_states)
    init_delta = utils.StartDistributionInitializer.stationary(init_gamma)
    return init_lambda, init_gamma, init_delta


def load_data(path: Union[str, Path], file: Union[str, Path],
              dtype: str = 'float128'):
    """
    Load test data from file.
    """
    path = Path(path)
    dfp = path.joinpath(file)
    return np.fromfile(str(dfp), dtype=dtype, sep='\n')
