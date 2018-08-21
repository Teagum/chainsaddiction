#ifndef FWBW_H
#define FWBW_H

#include "linalg.h"
#include "stats.h"

void fwbw(Vector *x, size_t m, size_t n, Vector *_lambda, Matrix *_gamma, Vector *_delta);

#endif	/* FWBW_H */
