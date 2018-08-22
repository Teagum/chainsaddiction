import numpy as np
from scipy.stats import poisson
from apollon.hmm.fwbw import forward_backward

m = 3
n = 5

x = np.array( [15, 24, 33, 42, 12, 13, 14], dtype=int)

lam = np.array([10, 20, 30], dtype=np.double)

gam = np.array([[.7, .2, .1],
                [.1, .7, .2],
                [.2, .1, .7]])

delta = np.array([.5, .3, .2])

# init alpha
alpha = np.zeros((n, m))

_at = poisson.pmf(x[0], lam) * delta
sat = _at.sum()
lsf = np.log(sat)
_at /= sat

alpha[0] = np.log(_at) + lsf
# start recursion


a, b = forward_backward(x, m, lam, gam, delta)
print(a)
print("\n\n")
print(b)
