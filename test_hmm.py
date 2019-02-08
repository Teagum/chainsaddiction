import numpy as np
import hmm

class PoissonHMM:
    def __init__(self, x, _lambda, _gamma, _delta, max_iter=1000, tol=1e-5):
        ret = hmm.hmm_poisson_EM(x, _lambda, _gamma, _delta, max_iter, tol)
        self.success = ret[0]
        self.lambda_ = ret[1]
        self.gamma_ = ret[2]
        self.delta_ = ret[3]
        self.aic = ret[4]
        self.bic = ret[5]
        self.nll = ret[6]
        self.n_iter = ret[7]

    def __str__(self):
        fmt = '\nSuccess: {}\n\nLambda:\n{}\n\nGamma:\n{}\n\nDelta:\n{}\n\nAIC: {}\nBIC: {}\nnll: {}\nn_iter: {}\n'
        return fmt.format(self.success, self.lambda_, self.gamma_, self.delta_, self.aic, self.bic,
                self.nll, self.n_iter)


x = np.array([  13, 14,  8, 10, 16, 26, 32, 27, 18, 32,
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

m = 3

_lambda = np.array([10., 20., 30.])

_gamma = np.array([[.8,  .1, .1],
                   [ .1, .8, .1],
                   [ .1, .1, .8]])

_delta = np.array([1./3., 1./3., 1./3.])


pmm = PoissonHMM(x, _lambda, _gamma, _delta)
print(pmm)


