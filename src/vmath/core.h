#ifndef vmath_core_h
#define vmath_core_h

#include "stdlib.h"
#include "config.h"


extern void
copy_op (
    const size_t n_elem,
    const size_t stride,
    scalar (*op) (scalar),
    const scalar *restrict vtx,
    scalar *restrict out);


extern void
inplace_op (
    const size_t n_elem,
    const size_t stride,
    scalar (*op) (scalar),
    scalar *restrict vtx);


extern void 
acc_sum (
    size_t n_elem,
    size_t stride,
    const scalar *restrict vtx,
    scalar *restrict res);


extern void
acc_prod (
    size_t n_elem,
    size_t stride,
    const scalar *restrict vtx,
    scalar *restrict res);


extern void
acc_sum_op (
    size_t n_elem,
    size_t stride,
    scalar (*op) (scalar),
    const scalar *restrict vtx,
    scalar *res);


extern void
acc_prod_op (
    size_t n_elem,
    size_t stride,
    scalar (*op) (scalar),
    const scalar *restrict vtx,
    scalar *res);


#endif  /* vmath_core_h */
