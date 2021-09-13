"""
ChaisAddiction testing utilities
"""
import numpy as np
from apollon.hmm import utilities as utils


def gen_init_params(m_states: int, data: np.ndarray) -> tuple:
    """Generate initila parameters for HMM training."""
    init_lambda = utils.StateDependentMeansInitializer.hist(data, m_states)
    init_gamma = utils.TpmInitializer.softmax(m_states)
    init_delta = utils.StartDistributionInitializer.stationary(init_gamma)
    return init_lambda, init_gamma, init_delta
