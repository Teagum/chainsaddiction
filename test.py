import hmm
import numpy as np
#import matplotlib.pyplot as plt

x = np.arange(1, 20, dtype=int)
m = 2
lam = np.array([9, 16], dtype=np.double)
gam = np.array([[.3, .7], [.8, .2]])
delta = np.array([.5, .5])

ff = hmm.poisson_fwbw(x, m, lam, gam, delta)

print(ff)
