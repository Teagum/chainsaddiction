#ifndef RND_H
#define RND_H

#include <stdlib.h>
#include <time.h>
#include "scalar.h"
#include "restrict.h"

/* Return random scalar.
 */
extern scalar
rnd
    (void);


/* Random integer in range.
 * @param _r_min - Lower bound, inclusive.
 * @param _r_max - Upper bound, exclusive.
 */
extern int
rnd_int (
    const int _r_min,
    const int _r_max);


/* Fill vector wiht random scalars
 *
 * @param n_elem
 * @param _buffer
 */
extern void
v_rnd (
    const size_t n_elem,
    scalar *restrict _buffer);


/* Fill vector with random integers in range.
 * @param _r_min - Lower bound, inclusive.
 * @param _r_max - Upper bound, exclusive.
 * @param _n_elem
 * @param _buffer
 */
extern void
v_rnd_int (
    const int _r_min,
    const int _r_max,
    const size_t _n_elem,
    int *restrict _buffer);

#endif  /* RND_H */
