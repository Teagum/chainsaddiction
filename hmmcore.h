#ifndef FWBW_H
#define FWBW_H

#include "linalg.h"
#include "stats.h"


typedef struct {
    Matrix *alpha;
    Matrix *beta;
    Matrix *prob;
} Lfbp;


typedef struct {
	Vector *lambda_;
	Matrix *gamma_;
	Vector *delta_;
} HmmParams;


Lfbp *lfwbw(const Scalar *x, const size_t n, const size_t m,
		const Scalar *_lambda, Scalar **_gamma, const Scalar *_delta);

HmmParams *EM(const Scalar *x, const size_t n, const size_t m,
	      	const size_t max_iter, const Scalar tol,	
		const Scalar *__lambda, Scalar **__gamma, const Scalar *__delta);

#endif	/* FWBW_H */
