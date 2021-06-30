#include "utils.h"


scalar
compute_aic(scalar nll, size_t m)
{
    return 2.0L * (scalar) (nll + 2*m + m*m);
}
