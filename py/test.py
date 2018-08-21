import numpy as np
from scipy.stats import poisson


x = np.array( [15, 24, 33, 42, 51], dtype=int)
m = 3
n = 5

lam = np.array([10, 20, 30], dtype=np.double)
gam = np.array([[.7, .2, .1],
                [.1, .7, .2],
                [.2, .1, .7]
                ])

delta = np.array([.5, .3, .2])


prob_i = poisson.pmf(x[0], lam)
alpha_i = delta * prob_i

sai = alpha_i.sum()
lsf = np.log1p(sai)

print("prob_i: {}".format(prob_i))
print("alpha_i: {} ".format(alpha_i))
print("sai: {}, lsf: {}".format(sai, lsf))

#ff = hmm.poisson_fwbw(x, m, lam, gam, delta)

