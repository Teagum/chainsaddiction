from dataclasses import dataclass
import unittest
import numpy as np
import chains_addiction as ca

""" test_em.py
Test cases for EM algorithm.
"""

Array = np.ndarray
earthquake = np.array([  13, 14,  8, 10, 16, 26, 32, 27, 18, 32,
                36, 24, 22, 23, 22, 18, 25, 21, 21, 14,
                 8, 11, 14, 23, 18, 17, 19, 20, 22, 19,
                13, 26, 13, 14, 22, 24, 21, 22, 26, 21,
                23, 24, 27, 41, 31, 27, 35, 26, 28, 36,
                39, 21, 17, 22, 17, 19, 15, 34, 10, 15,
                22, 18, 15, 20, 15, 22, 19, 16, 30, 27,
                29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
                18, 14, 10, 15,  8, 15,  6, 11,  8,  7,
                18, 16, 13, 12, 13, 20, 15, 16, 12, 18,
                15, 16, 13, 15, 16, 11, 11])

class HmmParams:
    m_states: int
    init_lambda: Array
    init_gamma: Array
    init_delta: Array
    max_iter: int
    tol: float

class TestEM(unittest.TestCase):
    def setUp(self):
        self.params = HmmParams(3, np.array([10, 250, 450]),
            np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]),


        self.params_random = {
            'x_train': np.random.randint(0, 500, 1000),
            'm_states': 3,
            'init_lambda': 
            'init_gamma': 
            'init_delta': np.array([.3, .3, .3]),
            'max_iter': 1000,
            'tol': 1e-6}

        self.params_earthquake = {
            'x_train': earthquake,
            'm_states': 3,
            'init_lambda': np.array([10., 20., 30.]),
            'init_gamma': np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]),
            'init_delta': np.array([1/3., 1/3., 1/3.]),
            'max_iter': 1000,
            'tol': 1e-6}

    def test_random(self):
        ph = PoissonHMM(**self.params_random)
        self.assertFalse(np.allclose(ph.gamma_, gamma_))

    def test_earthquake(self):
        ph = PoissonHMM(**self.params_earthquake)
        self.assertFalse(np.allclose(ph.gamma_, gamma_))

"""
class PoissonHMM:
    def __init__(self, x_train, m_states: int, init_lambda, init_gamma,
            init_delta, max_iter: int, tol: float):
        ret = ca.hmm_poisson_fit_em(x_train, m_states, init_lambda,
                init_gamma, init_delta, max_iter, tol)
        self.success = ret[0]
        self.lambda_ = ret[1]
        self.gamma_ = ret[2]
        self.delta_ = ret[3]
        self.aic = ret[4]
        self.bic = ret[5]
        self.nll = ret[6]
        self.n_iter = ret[7]

    def __str__(self):
        fmt = ('\nSuccess: {}\n\nLambda:\n{}\n\nGamma:\n{}\n\nDelta:\n{}\n\nAIC:'
            '{}\nBIC: {}\nnll: {}\nn_iter: {}\n')
        return fmt.format(self.success, self.lambda_, self.gamma_, self.delta_,
            self.aic, self.bic, self.nll, self.n_iter)
"""
