import numpy as np
from scipy.stats import poisson
from apollon.hmm.fwbw import forward_backward
from apollon.hmm.em import EM
from apollon.datasets import load_earthquakes

m = 3


eq = load_earthquakes()
x = eq.data[:14]
n = len(x)
print("n = {}".format(n));
lam = np.array([10, 20, 30], dtype=np.double)

gam = np.array([[.8, .1, .1],
                [.1, .8, .1],
                [.1, .1, .8]])

delta = np.array([.333333, .333333, .333333])


a, b, p = forward_backward(x, m, lam, gam, delta)

print(a)
#out = EM(x, 3, (lam, gam, delta))

#print(out[0], "\n")
#print(np.round(out[1],4), "\n")
#print(out[2])
