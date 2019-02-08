import numpy as np
import hmm

x = np.random.randint(0, 500, 1000)

l = np.array([10, 250, 450])
g = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
d = np.array([.3, .3, .3])


(s, l_, g_, d_, aic, bic, nll, niter) = hmm.hmm_poisson_EM(x, l, g, d, 1000, 1e-6)
print(s, aic, bic, nll, niter)

print(l_)
print("\n")
print(g_)
print("\n")
print(d_)
