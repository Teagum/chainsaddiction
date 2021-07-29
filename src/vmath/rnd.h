#ifndef rnd_h
#define rnd_h

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
/** Sample random integer from interval.
 *
 * \param[in] r_min    Lower bound, inclusive.
 * \param[in] r_max    Upper bound, exclusive.
 *
 * \return Random integer.
 */
extern int
rnd_int (
    const int r_min,
    const int r_max);


/* Sample `n_elem' random integers form interval.
 * \param[in]  n_elem   Number of elements.
 * \param[in]  r_min    Lower bound of sampling range.
 * \param[in]  r_max    Upper bound of sampling range.
 * \param[out] buffer   Pointer to allocated memory.
 */
extern void
v_rnd_int (
    const size_t n_elem,
    const int r_min,
    const int r_max,
    int *const restrict samples);


/** Sample random scalar form interval.
 *
 * \param[in] r_min     Lower bound of sampling range.
 * \param[in] r_max     Uppder bound of sampling range.
 *
 * \return Random scalar value.
 */
extern scalar
rnd_scalar (
    const scalar r_min,
    const scalar r_max);


/** Sample `n_elem` random scalars form interval.
 *
 * \param[in]  n_elem   Number of samples.
 * \param[in]  r_min    Lower bound of sampling range.
 * \param[in]  r_max    Uppder bound of sampling range.
 * \param[out] samples  Pointer to allocated memory.
 *
 */
extern void
v_rnd_scalar (
    const size_t n_elem,
    const scalar r_min,
    const scalar r_max,
    scalar *restrict samples);


/** Sample random scalar from unit interval.
 *
 * \return Random scalar from [0.0, 1.0].
 *
 */
#define sample() RND_SCALAR

/** Sample `n_elem random scalars from unit interval.
 *
 * \param[in] n_elem    Number of samples.
 *
 */
extern void
v_sample (
    size_t n_elem,
    scalar *restrict samples);


#define m_sample(n_rows, n_cols, buffer) v_sample (n_rows * n_cols, buffer)

#define m_rnd_size(r_min, r_max, n_rows, n_cols, buffer) \
    v_rnd_size (r_min, r_max, n_rows * n_cols, buffer)

#define m_rnd_scalar(r_min, r_max, n_rows, n_cols, buffer) \
    v_rnd_scalar (r_min, r_max, n_rows * n_cols, buffer)


#endif  /* rnd_h */
