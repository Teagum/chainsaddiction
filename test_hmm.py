import numpy as np
import hmm

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


y = hmm.hmm_poisson_EM(x, _lambda, _gamma, _delta, 1000, 1e-5)
if y[0] == 0:
    print('Model failed\n')
else:
    print('Success: {}\n'.format(y[0]))
    print('Lambda:\n{}\n'.format(y[1]))
    print('Gamma:\n{}\n'.format(y[2]))
    print('Delta:\n{}\n'.format(y[3]))
    print('AIC: {}\n'.format(y[4]))
    print('BIC: {}\n'.format(y[5]))
    print('nLL: {}\n'.format(y[6]))
    print('iter: {}\n'.format(y[7]))

