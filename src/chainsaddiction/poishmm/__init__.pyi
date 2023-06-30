from typing import Any
from .. typing import AnyArray


class PoissonHmm:
    ...

def fit(n_obs: int, m_states: int, max_iter: int, tol: float, sdm: AnyArray,
        tpm: AnyArray, distr: AnyArray, data: AnyArray) -> PoissonHmm:
    ...

def read_params(path: str) -> dict[str, Any]:
    ...
