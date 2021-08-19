#ifndef VMATH_H
#define VMATH_H

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "alloc.h"
#include "config.h"
#include "core.h"
#include "print.h"


enum vmath_error_codes {
    VM_ERR_ZERO_SIZED_BUFFER = -1,
};


#define FOR_EACH(idx, max) for (size_t (idx) = 0; (idx) < (max); (idx++))

#define OUTER_LOOP for (size_t i = 0; i < n_elem; i++)
#define INNER_LOOP for (size_t j = 0; j < n_elem; j++)


/*
 * ============================================================================
 * Vector interface
 * ============================================================================
 */

/** Operations on vectors
 *
 * Evaluate function elementwise and copy result to output buffer.
 *
 * \param[in]   n_elem      Number of elements in vector.
 * \param[in]   vtx         Pointer to vector data.
 * \param[out]  out         Pointer to output obejct.
 */
extern scalar   v_sum       (size_t n_elem, const scalar *restrict vtx);
extern scalar   v_sumlog    (size_t n_elem, const scalar *restrict vtx);
extern scalar   v_sumexp    (size_t n_elem, const scalar *restrict vtx);
extern void     v_exp       (size_t n_elem, const scalar *restrict vtx, scalar *restrict out);
extern void     v_log       (size_t n_elem, const scalar *restrict vtx, scalar *restrict out);
extern void     v_logr1     (size_t n_elem, const scalar *restrict vtx, scalar *restrict out);
extern scalar   v_lse       (size_t n_elem, const scalar *restrict vtx);
extern scalar   v_max       (size_t n_elem, const scalar *restrict vtx);
extern scalar   v_min       (size_t n_elem, const scalar *restrict vtx);
extern size_t   v_argmax    (size_t n_elem, const scalar *restrict vtx);
extern size_t   v_argmin    (size_t n_elem, const scalar *restrict vtx);
extern scalar   v_acc_prod  (size_t n_elem, size_t stride, scalar (*op) (scalar), const scalar *restrict vtx);
extern scalar   v_acc_sum   (size_t n_elem, size_t stride, scalar (*op) (scalar), const scalar *restrict vtx);

#define def_v_op(name, op_func)                                     \
inline void v_##name                                                \
(size_t n_elem, const scalar *restrict vtx, scalar *restrict out)   \
{                                                                   \
    copy_op (n_elem, 1, op_func, vtx, out);                         \
}


/** Inplace operations on vectors
 *
 * Evaluate function elementwise and copy result to output buffer.
 *
 * \param[in]   n_elem      Number of elements in vector.
 * \param[in]   vtx         Pointer to vector data.
 */
extern void vi_exp   (size_t n_elem, scalar *restrict vtx);
extern void vi_log   (size_t n_elem, scalar *restrict vtx);
extern void vi_logr1 (size_t n_elem, scalar *restrict vtx);

#define def_vi_op(name, op_func) \
inline void                                         \
vi_##name (size_t n_elem, scalar *restrict vtx) {   \
    inplace_op (n_elem, 1, op_func, vtx);           \
}


/*
 * ============================================================================
 * Vector/scalar interface
 * ============================================================================
 */

/** Basic vector/scalar arithmetic
 *
 * Perform given operation and copy the result to `out`.
 *
 * \param[in]   n_elem      Number of elemets.
 * \param[in]   alpha       Constant scalar value.
 * \param[in]   vtx         Pointer to vector data.
 * \param[out]  out         Pointer to output object.
 */
extern void vs_add (size_t n_elem, const scalar alpha, scalar *vtx, scalar *out);
extern void vs_sub (size_t n_elem, const scalar alpha, scalar *vtx, scalar *out);
extern void vs_mul (size_t n_elem, const scalar alpha, scalar *vtx, scalar *out);
extern void vs_div (size_t n_elem, const scalar alpha, scalar *vtx, scalar *out);

#define def_vs_op(name, op)         \
inline void vs_##name (             \
    size_t n_elem,                  \
    const scalar alpha,             \
    scalar *vtx,                    \
    scalar *out)                    \
{                                   \
    while (n_elem--) {              \
        *out++ = alpha op *vtx++;   \
    }                               \
}


/** Basic vector/scalar inplace arithmetic
 *
 * Perform given operation inplace
 *
 * \param[in]   n_elem      Number of elemets.
 * \param[in]   alpha       Constant scalar value.
 * \param[in]   vtx         Pointer to vector data.
 */
extern void vsi_add (size_t n_elem, const scalar alpha, scalar *restrict vtx);
extern void vsi_sub (size_t n_elem, const scalar alpha, scalar *restrict vtx);
extern void vsi_mul (size_t n_elem, const scalar alpha, scalar *restrict vtx);
extern void vsi_div (size_t n_elem, const scalar alpha, scalar *restrict vtx);

#define def_vsi_op(name, op)        \
inline void vsi_##name (            \
    size_t n_elem,                  \
    const scalar alpha,             \
    scalar *restrict vtx)           \
{                                   \
    while (n_elem--) {              \
        *vtx++ op##= alpha;         \
    }                               \
}


/*
 * ============================================================================
 * Vector/vector interface
 * ============================================================================
 */


/** Basic vector/vector arithmetic
 *
 * Perform given operation and copy data to output buffer.  Vectors are
 * expected to be contiguous objects with a size of at least `n_elem * sizeof
 * (scalar)`.
 *
 * \param[in]   n_elem      Number of elements in each vector.
 * \param[in]   vtx         Pointer to vector data.
 * \param[in]   vty         Pointer to vector data.
 * \param[out]  out         Pointer to output object.
 */
extern void vv_add (size_t n_elem, scalar *vtx, scalar *vty, scalar *out);
extern void vv_sub (size_t n_elem, scalar *vtx, scalar *vty, scalar *out);
extern void vv_mul (size_t n_elem, scalar *vtx, scalar *vty, scalar *out);
extern void vv_div (size_t n_elem, scalar *vtx, scalar *vty, scalar *out);

#define def_vv_op(name, op)                                                 \
inline void vv_##name (size_t n_elem, scalar *vtx, scalar *vty, scalar *out)\
{                                                                           \
    while (n_elem--) {                                                      \
        *out++ = *vtx++ op *vty++;                                          \
    }                                                                       \
}


/** Basic vector/vector inplace arithmetic
 *
 * Perform given operation and modify data of second buffer.  Vectors are
 * expected to be contiguous objects with a size of at least `n_elem * sizeof
 * (scalar)`.
 *
 * \param[in]       n_elem      Number of elements in each vector.
 * \param[in]       vtx         Pointer to vector data.
 * \param[in, out]  vty         Pointer to vector data.
 */
extern void vvi_add (size_t n_elem, scalar *vtx, scalar *vty);
extern void vvi_sub (size_t n_elem, scalar *vtx, scalar *vty);
extern void vvi_mul (size_t n_elem, scalar *vtx, scalar *vty);
extern void vvi_div (size_t n_elem, scalar *vtx, scalar *vty);

#define def_vvi_op(name, op)                                        \
inline void vvi_##name (size_t n_elem, scalar *vtx, scalar *vty)    \
{                                                                   \
    while (n_elem--) {                                              \
        *vty++ op##= *vtx++;                                        \
    }                                                               \
}


#define def_mm_op_s_func(name, op)          \
inline void                                 \
mm_ ## name ##_s (                          \
    const scalar *mta,                      \
    const scalar *mtb,                      \
    const size_t n_elem,                    \
    const scalar val,                       \
    scalar *restrict buffer)                \
{                                           \
    OUTER_LOOP                              \
    {                                       \
        buffer[i] = mta[i] + mtb[i] + val;  \
    }                                       \
}




/** Compute basic matrix/matrix operations with added constant.
 *
 * \param mat       Pointer the first matrix data.
 * \param mtb       Pointer the second matrix data.
 * \param n_rows    Number of rows of each matrix.
 * \param n_cols    Numnber of columns in each matrix.
 * \param val       Constant value.
 * \param buffer    Pointer to output buffer.
 */
extern void
mm_add_s (
    const scalar *mta,
    const scalar *mtb,
    const size_t n_elem,
    const scalar val,
    scalar *restrict buffer);


extern void
mm_sub_s (
    const scalar *mta,
    const scalar *mtb,
    const size_t n_elem,
    const scalar val,
    scalar *restrict buffer);


extern void mm_mul_s (
    const scalar *mta,
    const scalar *mtb,
    const size_t n_elem,
    const scalar val,
    scalar *restrict buffer);


extern void mm_div_s (
    const scalar *mta,
    const scalar *mtb,
    const size_t n_elem,
    const scalar val,
    scalar *restrict buffer);


/** Logarithm of the sum of the exponential of the vector elements.
 *
 * \param vctr      Pointer to input data.
 * \param n_elem    Number of vector elements.
 */


extern scalar
vs_lse_centroid (
    const scalar *restrict vt,
    const size_t v_stride,
    const scalar *restrict weights,
    const size_t w_stride,
    const size_t n_elem);

#define v_lse_centroid(vt, weights, n_elem) \
    vs_lse_centroid (vt, 1, weights, 1, n_elem)


extern size_t
v_argmax (const size_t n_elem, const scalar *restrict vec);


/** Compute softmax o `buffer' inplace.
 *
 * \param buffer    Pointer to object.
 * \param n_elem    Number of elements in object.
 */
extern void
vi_softmax (scalar *buffer, size_t n_elem);


/* === Strided vector interface ==== */

/** Compute the sum of the vector elements given a stride.
 *
 * \param _vt - Pointer to input data.
 * \param n_elem - Number of elements.
 * \param stride - Stride.
 */
extern scalar
vs_sum (
    const scalar *restrict _vt,
    const size_t n_elem,
    const size_t stride);


/* =====================
 * Matrix interface
 * ====================
 */

/** Compute column-wise weighted sum in log domain.
 *
 * `mtx' is a matrix of shape `n_rows' x `n_cols'.
 * `wgt' is a weight vector of size `n_rows' that is applied to
 * each column of `mtx'.
 *
 * This function computes the centroids using the LSE method.
 *
 * \param[in]  mtx      Pointer to matrix data.
 * \param[in]  wgt      Pointer to weight data.
 * \param[in]  n_rows   Number of matrix rows.
 * \param[in]  n_cols   Number of matrix columns.
 * \param[out] centroid_in_col      Output buffer.
 */
extern int
m_log_centroid_cols(
        const scalar *restrict mtx,
        const scalar *restrict wgt,
        const size_t n_rows,
        const size_t n_cols,
        scalar *const centroid_in_col);


#define m_argmax(rows, cols, mtx) v_argmax(((rows) * (cols)), mtx)


extern void
m_row_argmax (
    const size_t rows,
    const size_t cols,
    const scalar *mtx,
    size_t *restrict argmax_row);


/** Compute maximum value.
 *
 * \param _mat
 * \param _n_rows
 * \param _n_cols
 */
extern scalar
m_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols);


/** Compute maximum along rows.
 *
 * \param _mat
 * \param _n_rows
 * \param _n_cols
 * \paran _row_max
 */
extern void
m_row_max (
    const scalar *restrict mt,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict row_max);


/** Compute maximum along the matrix columns.
 *
 * \param mtx           Pointer to matrix buffer.
 * \param n_rows        Number of rows.
 * \param n_cols        Number of columns.
 * \paran max_per_col   Pointer to output object.
 *
 * Note: If `n_rows' and `n_cols' are zero, the function aborts and returns
 *       VM_ERR_ZERO_SIZED_BUFFER. The output buffer is not written.
 */
extern int
m_col_max (
    const scalar *restrict mtx,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict max_per_col);


/** Compute absolute maximum along the matrix columns.
 *
 * \param mtx           Pointer to matrix buffer.
 * \param n_rows        Number of rows.
 * \param n_cols        Number of columns.
 * \paran max_per_col   Pointer to output object.
 *
 * Note: If `n_rows' and `n_cols' are zero, the function aborts and returns
 *       VM_ERR_ZERO_SIZED_BUFFER. The output buffer is not written.
 */
extern int
m_col_absmax (
    const scalar *restrict mtx,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict max_per_col);


/** Compute exponentials of matrix elements.
 *
 * \param rows  Number of rows in matrix.
 * \param cols  Number of cols in matrix.
 * \param mtx   Pointer to matrix data.
 * \param out   Pointer to output buffer.
 */
#define m_exp(rows, cols, mtx, out) v_exp ((mtx), (rows*cols), (out))


/** Compute logarithm of matrix elements.
 *
 * \param mat       Pointer to matrix elements.
 * \param n_elem    Number of matrix elements.
 * \param out       Pointer to output buffer.
 */
#define m_log(mat, n_elem, out) v_log ((mat), (n_elem), (out))


/** Compute exponentials of matrix elements inplace.
 *
 * \param rows  Number of rows in matrix.
 * \param cols  Number of cols in matrix.
 * \param mtx   Pointer to matrix data.
 */
#define mi_exp(rows, cols, mtx) vi_log ((mtx), (rows*cols))


/** Compute logarithm of matrix elements inplace
 *
 * \param mat       Pointer to matrix elements.
 * \param n_elem    Number of matrix elements.
 */
#define mi_log(mat, n_elem) vi_log ((mat), (n_elem))


/** Compute vector/matrix product in log domain.
 *
 * Compute the product of a vector and a square matrix with the same
 * number of elements in each row and column. Computation is performed
 * in log domain by means of LSE.
 *
 * \param _vt    - Pointer to vector elements.
 * \param _mt    - Pointer to matrix elements.
 * \param n_elem - Number of elements in _vt.
 * \param _cs    - Computation buffer of lenght n_elem.
 * \param _mb    _ Computation buffer of lenght n_elem^2.
 * \param prod   - Output buffer of lenght n_elem.
 */
extern void
log_vmp (
    const scalar *restrict _vt,
    const scalar *restrict _mat,
    const size_t n_elem,
    scalar *_cs,
    scalar *_mb,
    scalar *_prod);


/** Compute matrix/vector product in log domain.
 *
 * Compute the product of a square matrix with n_elem rows and columns and a
 * vector with n_elem elements. Computation is performed in log domain by
 * means of LSE.
 *
 * \param _mt    - Pointer to matrix elements.
 * \param _vt    - Pointer to vector elements.
 * \param n_elem - Number of elements in _vt.
 * \param _cs    - Computation buffer of lenght n_elem.
 * \param _mb    _ Computation buffer of lenght n_elem^2.
 * \param prod   - Output buffer of lenght n_elem.
 */
extern void
log_mvp (
    const scalar *restrict _mt,
    const scalar *restrict _vt,
    const size_t n_elem,
    scalar *_cs,
    scalar *_mb,
    scalar *_prod);

/*
 * Private API
 */

/* Compute the maximum value along a strided buffer
 *
 * \param _buffer - Input buffer;
 * \param _n_elem - Number of elements in buffer.
 * \param _stride - Stride in elements.
 */
extern scalar
strided_max (
    const size_t n_elem,
    const size_t stride,
    const scalar *restrict buffer);


extern scalar
strided_absmax (
    const size_t n_elem,
    const size_t stride,
    const scalar *restrict buffer);



extern scalar logr1 (scalar val);


#endif  /* VMATH_H */
