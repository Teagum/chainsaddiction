#ifndef HMM_H
#define HMM_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scalar.h"

typedef struct {
    scalar *restrict lambda;
    scalar *restrict gamma;
    scalar *restrict delta;
} PoisParams;

typedef struct {
    size_t m;
    size_t n_iter;
    size_t max_iter;
    scalar tol;
    scalar aic;
    scalar bic;
    scalar nll;
    PoisParams *init;
    PoisParams *params;
} PoisHmm;

/** Allocate a new Params struct. Elements of
 * parameter vectors remain uninitialized.
 */
PoisParams *PoisHmm_NewEmptyParams (size_t m);

/** Allocate new PoisParams and init with parameters read from file. */
PoisParams *PoisHmm_ParamsFromFile (const char *fname);

/** Print Poisson parameters to stdout. */
void PoisHmm_PrintParams (PoisParams *params, size_t m_states);

/** Deallocate the Parms struct */
void PoisHmm_FreeParams (PoisParams *params);

/** Allocate new PoisHmm with init data from compile time constants. */
PoisHmm *
PoisHmm_FromData(size_t  m,
        scalar *restrict init_lambda,
        scalar *restrict init_gamma,
        scalar *restrict init_delta,
        size_t max_iter,
        scalar tol);

/** Deallocate a PoisHmm.*/
void
PoisHmm_DeleteHmm (PoisHmm *ph);

/* Compute Akaine Information criterion. */
scalar
compute_aic(scalar nll, size_t m);

/* Compute Bayes Information criterion. */
scalar
compute_bic(scalar nll, size_t m, size_t n);

#endif  /* HMM_H */
