#ifndef FWBW_H
#define FWBW_H

#include "linalg.h"
#include "stats.h"


typedef struct {
    Matrix *alpha;
    Matrix *beta;
} AB_probs;

AB_probs *fwbw(Scalar *x, size_t n, size_t m, Scalar *_lambda, Scalar **_gamma, Scalar *_delta);

#endif	/* FWBW_H */
