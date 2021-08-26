#ifndef rnd_h
#define rnd_h

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include "config.h"


/*
 * Basic macros
 */
#define SCALAR_RAND ((scalar) rand ())
#define RND_INT     (rand ())
#define RND_SIZE    ((size_t) rand ())
#define RND_SCALAR  (SCALAR_RAND / RAND_MAX)

#define ITER(x_n_elem, x_buffer) \
    for (size_t i = 0; i < x_n_elem; i++, x_buffer++)


/** Random scalars
 *
 * Sample single random value of given type from from interval. In case of
 * `rnd_sample`, the interval is set to the unit interval [0, 1].
 *
 * \param[in]   r_min   Lower bound of sampling range.
 * \param[in]   r_max   Upper bound of sampling range.
 *
 * \return Random integer within [r_min, r_max].
 */
extern int
rnd_int (const int r_min, const int r_max);

extern size_t
rnd_size (const size_t r_min, const size_t r_max);

extern scalar
rnd_scalar (const scalar r_min, const scalar r_max);

extern scalar
rnd_sample (void);


/** Random vectors
 *
 * Sample a vector of `n_elem' random values of given type from interval. In
 * case of `v_sample`, the interval is set to the unit interval [0, 1].
 *
 * `r_min` must be less then `r_max`. Otherwise the functions set errno
 * and return 0.
 *
 * \param[in]   n_elem      Number of elements.
 * \param[in]   r_min       Lower bound of sampling range.
 * \param[in]   r_max       Upper bound of sampling range.
 * \param[out]  samples     Pointer to allocated memory.
 */
extern void
v_rnd_int (
    const size_t n_elem,
    const int r_min,
    const int r_max,
    int *const restrict samples);

extern void
v_rnd_size (
    const size_t n_elem,
    const size_t r_min,
    const size_t r_max,
    size_t *restrict samples);

extern void
v_rnd_scalar (
    const size_t n_elem,
    const scalar r_min,
    const scalar r_max,
    scalar *restrict samples);

extern void
v_rnd_sample (
    size_t n_elem,
    scalar *restrict samples);


/** Random matrices
 *
 * Sample a matrix of `rows` times `cols`  random values of given type from
 * interval. In case of `m_sample`, the interval is set to the unit interval
 * [0, 1].
 *
 * `r_min` must be less then `r_max`. Otherwise the functions set errno
 * and return 0.
 *
 * \param[in]   rows        Number of rows.
 * \param[in]   cols        Number of cols.
 * \param[in]   r_min       Lower bound of sampling range.
 * \param[in]   r_max       Upper bound of sampling range.
 * \param[out]  samples     Pointer to output object.
 */
#define m_rnd_int(n_rows, n_cols, r_min, r_max, samples) \
    v_rnd_size (n_rows * n_cols, r_min, r_max, samples)

#define m_rnd_size(n_rows, n_cols, r_min, r_max, samples) \
    v_rnd_size (n_rows * n_cols, r_min, r_max, samples)

#define m_rnd_scalar(n_rows, n_cols, r_min, r_max, samples) \
    v_rnd_scalar (n_rows * n_cols, r_min, r_max, samples)

#define m_rnd_sample(n_rows, n_cols, samples) \
    v_rnd_sample (n_rows * n_cols, samples)


#endif  /* rnd_h */
