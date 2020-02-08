import unittest
import numpy as np
import chains_addiction as ca

""" test_em.py
Test cases for EM algorithm.
"""

class TestEM(unittest.TestCase):
    def setUp(self):
        self.params = {
            'x_train': np.random.randint(0, 500, 1000),
            'm_states': 3,
            'init_lambda': np.array([10, 250, 450]),
            'init_gamma': np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]),
            'init_delta': np.array([.3, .3, .3]),
            'max_iter': 1000,
            'tol': 1e-6
            }

    def test_bug(self):
        res = ca.hmm_poisson_fit_em(*self.params.values())
        (success, lambda_, gamma_, delta_, aic, bic, nll, niter) = res
        self.assertFalse(np.allclose(self.params['init_gamma'], gamma_))
