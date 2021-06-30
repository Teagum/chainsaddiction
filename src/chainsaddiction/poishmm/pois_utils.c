#include "pois_utils.h"


scalar
compute_aic(scalar nll, size_t m)
{
    return 2.0L * (scalar) (nll + 2*m + m*m);
}


scalar
compute_bic(scalar nll, size_t m, size_t n)
{
    return 2.0L * nll + logl ((scalar) n) * (scalar) (2*m + m*m);
}
