import numpy as np
import hmm

x = np.random.randint(0, 500, 1000)
m = 3
l = np.array([10, 250, 450])
g = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
d = np.array([.3, .3, .3])


(success, lambda_, gamma_, delta_, aic, bic, nll, niter) = hmm.hmm_poisson_fit_em(x, m, l, g, d, 1000, 1e-6)
print(success, aic, bic, nll, niter)

print(lambda_)
print("\n")
print(gamma_)
print("\n")
print(delta_)
