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


/* Sample random scalar from the unit interval.
 */
extern scalar
rnd (void);


/** Sample `n_elem` random scalars for the unit interval.
 *
 * \param[in]  n_elem    Number of elements in vector.
 * \param[out] buffer    Pointer to allocate memory.
 */
extern void
v_rnd (
    const size_t n_elem,
    scalar *const restrict buffer);


/** Sample random scalar form interval.
 *
 * \param[in] r_min     Lower bound of sampling range.
 * \param[in] r_max     Uppder bound of sampling range.
 *
 * \return Random scalar value.
 */
extern scalar
r_rnd (
    const scalar r_min,
    const scalar r_max);


/** Sample `n_elem` random scalars form interval.
 *
 * \param[in]  n_elem   Number of samples.
 * \param[in]  r_min    Lower bound of sampling range.
 * \param[in]  r_max    Uppder bound of sampling range.
 * \param[out] samples
 *
 */
extern void
vr_rnd (
    const size_t n_elem,
    const scalar r_min,
    const scalar r_max,
    scalar *restrict samples);


#define m_rnd(n_rows, n_cols, buffer) v_rnd (n_rows * n_cols, buffer)


#endif  /* rnd_h */
