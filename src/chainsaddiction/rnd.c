#include "rnd.h"

inline scalar
rnd
    (void)
{
    return (scalar) rand() / RAND_MAX;
}


inline int
rnd_int (
    const int _r_min,
    const int _r_max)
{
    return _r_min + (rand () % (_r_max - _r_min));
}


inline void
v_rnd
    (const size_t _n_elem,
     scalar *restrict _buffer)
{
    for (size_t i = 0; i < _n_elem; i++, _buffer++)
    {
        *_buffer = rnd ();
    }
}


inline void
v_rnd_int (
    const int _r_min,
    const int _r_max,
    const size_t _n_elem,
    int *restrict _buffer)
{
    for (size_t i = 0; i < _n_elem; i++, _buffer++)
    {
        *_buffer = rnd_int (_r_min, _r_max);
    }
}
