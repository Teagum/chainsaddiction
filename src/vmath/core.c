#include "core.h"


inline void
copy_op (const size_t n_elem, const size_t stride, scalar (*op) (scalar),
         const scalar *restrict vtx, scalar *restrict out)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *out = op (*vtx);
        out+=stride;
        vtx+=stride;
    }
}


inline void
inplace_op (const size_t n_elem, const size_t stride, scalar (*op) (scalar),
            scalar *restrict vtx)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *vtx = op (*vtx);
        vtx+=stride;
    }
}


inline void
acc_sum_op (size_t n_elem, size_t stride, scalar (*op) (scalar),
            const scalar *restrict vtx, scalar *restrict res)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *res += op (*vtx);
        vtx+=stride;
    }
}


inline void
acc_sum (size_t n_elem, size_t stride, const scalar *restrict vtx,
         scalar *restrict res)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *res += (*vtx);
        vtx+=stride;
    }
}


inline void
acc_prod_op (size_t n_elem, size_t stride, scalar (*op) (scalar),
             const scalar *restrict vtx, scalar *restrict res)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *res *= op (*vtx);
        vtx+=stride;
    }
}


inline void
acc_prod (size_t n_elem, size_t stride, const scalar *restrict vtx,
          scalar *restrict res)
{
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        *res += (*vtx);
        vtx+=stride;
    }
}
