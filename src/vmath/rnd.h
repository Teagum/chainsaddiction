#ifndef rnd_h
#define rnd_h

#include <stdlib.h>
#include <time.h>
#include "config.h"


#define SCALAR_RAND (scalar) rand ()


/* Sample random scalar from the unit interval.
 */
extern scalar
rnd (void);


/** Sample random scalar form interval.
 *
 * \param[in] r_min
 * \param[in] r_max
 *
 * \return Random scalar value.
 */
extern scalar
r_rnd (
    const scalar r_min,
    const scalar r_max);


/* Random integer in range.
 * \param r_min    Lower bound, inclusive.
 * \param r_max    Upper bound, exclusive.
 */
extern int
rnd_int (
    const int r_min,
    const int r_max);


/* Fill vector wiht random scalars
 *
 * \param[in]  n_elem    Number of elements in vector.
 * \param[out] buffer    Pointer to allocate memory.
 */
extern void
v_rnd (
    const size_t n_elem,
    scalar *const restrict buffer);


/* Fill vector with random integers in range.
 * \param[in]  r_min    Lower bound, inclusive.
 * \param[in]  r_max    Upper bound, exclusive.
 * \param[in]  n_elem   Number of elements.
 * \param[out] buffer   Pointer to allocated memory.
 */
extern void
v_rnd_int (
    const int r_min,
    const int r_max,
    const size_t n_elem,
    int *const restrict buffer);


#define m_rnd(n_rows, n_cols, buffer) v_rnd (n_rows * n_cols, buffer)


#endif  /* rnd_h */
