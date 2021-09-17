#ifndef pois_em_h
#define pois_em_h

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "../dataset.h"
#include "../restrict.h"
#include "../scalar.h"
#include "pois-em.h"
#include "pois-fwbw.h"
#include "pois-hmm.h"
#include "pois-params.h"
#include "pois-probs.h"


int
pois_em (
    const size_t n_obs,
    const size_t m_states,
    const size_t iter_max,
    const scalar tol,
    const scalar *const restrict data,
    size_t *restrict n_iter,
    scalar *restrict llh,
    PoisProbs *const restrict probs,
    PoisParams *const restrict params);


void
pois_e_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict input_data,
    const scalar *const restrict lambda,
    const scalar *const restrict lgamma,
    const scalar *const restrict ldelta,
    scalar *const restrict lsdp,
    scalar *const restrict lalpha,
    scalar *const restrict lbeta,
    scalar *const restrict lcsp,
    scalar *const restrict llh);


void
pois_m_step (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict data,
    const scalar *const restrict lsdp,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar *const restrict lcsp,
    const scalar *const restrict lgamma,
    scalar *const restrict new_lambda,
    scalar *const restrict new_lgamma,
    scalar *const restrict new_ldelta);


void
pois_m_step_lambda (
    const size_t n_obs,
    const size_t m_states,
    const scalar *const restrict data,
    const scalar *const restrict lcsp,
    scalar *const restrict new_lambda);


void
pois_m_step_gamma (
    const size_t n_obs,
    const size_t m_states,
    const scalar llh,
    const scalar *const restrict lsdp,
    const scalar *const restrict lalpha,
    const scalar *const restrict lbeta,
    const scalar *const restrict lgamma,
    scalar *const restrict new_lgamma);


void
pois_m_step_delta (
    const size_t m_states,
    const scalar *const restrict lcsp,
    scalar *const restrict new_ldelta);


scalar
score_update (
    const PoisParams *const restrict new,
    const PoisParams *const restrict old);


#endif  /* pois_em */
