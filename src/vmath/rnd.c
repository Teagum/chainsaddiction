#include "rnd.h"


inline scalar
rnd (void)
{
    return SCALAR_RAND / RAND_MAX;
}


inline int
rnd_int (const int r_min, const int r_max)
{
    return r_min + (rand () % (r_max - r_min));
}


inline void
v_rnd
    (const size_t n_elem,
     scalar *restrict buffer)
{
    for (size_t i = 0; i < n_elem; i++, buffer++)
    {
        *buffer = rnd ();
    }
}


inline void
v_rnd_int (
    const int r_min,
    const int r_max,
    const size_t n_elem,
    int *restrict buffer)
{
    for (size_t i = 0; i < n_elem; i++, buffer++)
    {
        *buffer = rnd_int (r_min, r_max);
    }
}
