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
    ITER (n_elem, samples)
    {
        *samples = rnd_int (r_min, r_max);
    }
}


inline scalar
rnd_scalar (const scalar r_min, const scalar r_max)
{
    return r_min + (SCALAR_RAND * (scalar)(r_max - r_min)) / RAND_MAX;
}


inline void
v_rnd_scalar (
    const size_t n_elem,
    const scalar r_min,
    const scalar r_max,
    scalar *restrict samples)
{
    ITER (n_elem, samples)
    {
        *samples = rnd_scalar (r_min, r_max);
    }
}


inline void
v_sample (
    const size_t n_elem,
    scalar *restrict samples)
{
    ITER (n_elem, samples)
    {
        *samples = sample ();
    }
}
