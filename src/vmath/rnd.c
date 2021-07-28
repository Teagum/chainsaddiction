#include "rnd.h"


inline int
rnd_int (const int r_min, const int r_max)
{
    return r_min + (RND_INT % (r_max - r_min));
}


inline void
v_rnd_int (
    const size_t n_elem,
    const int r_min,
    const int r_max,
    int *restrict samples)
{
    for (size_t i = 0; i < n_elem; i++, samples++)
    {
        *samples = rnd_int (r_min, r_max);
    }
}


inline scalar
rnd (void)
{
    return SCALAR_RAND / RAND_MAX;
}


inline scalar
r_rnd (const scalar r_min, const scalar r_max)
{
    return r_min + (SCALAR_RAND * (scalar)(r_max - r_min)) / RAND_MAX;
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
vr_rnd (
    const size_t n_elem,
    const scalar r_min,
    const scalar r_max,
    scalar *restrict samples)
{
    for (size_t i = 0; i < n_elem; i++, samples++)
    {
        *samples = r_rnd (r_min, r_max);
    }
}
