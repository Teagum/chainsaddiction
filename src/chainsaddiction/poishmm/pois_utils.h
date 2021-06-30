#ifndef pois_utils_h
#define pois_utils_h

#include <math.h>
#include <stdlib.h>
#include "../scalar.h"


/* Compute Akaine Information criterion. */
scalar
compute_aic(scalar nll, size_t m);


/* Compute Bayes Information criterion. */
scalar
compute_bic(scalar nll, size_t m, size_t n);


#endif  /* pois_utils.h */
