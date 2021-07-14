#ifndef RND_H
#define RND_H

#include <stdlib.h>
#include <time.h>
#include "config.h"

/* Return random scalar.
 */
extern scalar
rnd
    (void);


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
 * \param n_elem    Number of elements in vector.
 * \param buffer    Pointer to allocate memory.
 */
extern void
v_rnd (
    const size_t n_elem,
    scalar *restrict buffer);


/* Fill vector with random integers in range.
 * \param r_min    Lower bound, inclusive.
 * \param r_max    Upper bound, exclusive.
 * \param n_elem   Number of elements.
 * \param buffer   Pointer to allocated memory.
 */
extern void
v_rnd_int (
    const int r_min,
    const int r_max,
    const size_t n_elem,
    int *restrict buffer);

#endif  /* RND_H */
