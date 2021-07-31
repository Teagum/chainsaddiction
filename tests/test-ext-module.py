import numpy as np
from chainsaddiction import hmm_poisson_fit_em

earthquakes = np.fromfile('tests/data/earthquakes', sep='\n')
centroids = np.fromfile('tests/data/centroids', sep='\n')

params_4s = {
    'n_obs': centroids.size,
    'm_states': 4,
    'max_iter': 100,
    'tol': 1e-5,
}

hmm_params_4s = {
    'lambda': np.array([100, 1000, 5000, 10000]),
    'gamma': np.array([[0.7, 0.1, 0.1, 0.1],
                       [0.1, 0.7, 0.1, 0.1],
                       [0.1, 0.1, 0.7, 0.1],
                       [0.1, 0.1, 0.1, 0.7]]),
    'delta': np.array([1/4, 1/4, 1/4, 1/4]),
}

params_3s = {
    'n_obs': earthquakes.size,
    'm_states': 3,
    'max_iter': 100,
    'tol': 1e-5,
}

hmm_params_3s = {
    'lambda': np.array([10, 20, 30]),
    'gamma': np.array([[0.8, 0.1, 0.1],
                       [0.1, 0.8, 0.1],
                       [0.1, 0.1, 0.8]]),
    'delta': np.array([1, 0, 0]),
}

print("Earthquakes 3 states:")
hmm_poisson_fit_em(*params_3s.values(), *hmm_params_3s.values(), earthquakes)

print("\nCentroids 4 states:")
hmm_poisson_fit_em(*params_4s.values(), *hmm_params_4s.values(), centroids)
