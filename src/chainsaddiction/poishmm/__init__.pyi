from typing import Any
from .. typing import AnyArray, Float128Array


class PoisHmm:
    err: bool
    n_iter: int
    m_states: int
    aic: float
    bic: float
    llk: float
    lambda_: Float128Array
    gamma_: Float128Array
    delta_: Float128Array
    lalpha: Float128Array
    lbeta: Float128Array
    lcsp: Float128Array


def fit(n_obs: int, m_states: int, max_iter: int, tol: float, sdm: AnyArray,
        tpm: AnyArray, distr: AnyArray, data: AnyArray) -> PoisHmm:
    ...

def read_params(path: str) -> dict[str, Any]:
    ...
