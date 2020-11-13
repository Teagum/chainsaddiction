#ifndef VMATH_H
#define VMATH_H

#include <math.h>
#include <stdlib.h>
#include "restrict.h"
#include "scalar.h"
#include "utilities.h"

/*
 * Prefixes:
 * v:   vector operation
 * m:   matrix operation
 * i:   inplace operation
 * s:   strided operation
 */

/*
 * Postfixes:
 * s:   scalar
 */
#define OUTER_LOOP for (size_t i = 0; i < n_elem; i++)
#define INNER_LOOP for (size_t j = 0; j < n_elem; j++)

#define M_OUTER_LOOP for (size_t i = 0; i < _n_rows; i++)
#define M_INNER_LOOP for (size_t j = 0; j < _n_cols; j++)
#define ITER_MATRIX \
   M_OUTER_LOOP { \
      M_INNER_LOOP { \
#define END_ITER_MATRIX }}


#define def_vi_s_func(name, op)     \
inline void                         \
vi_ ## name ##_s (                  \
    scalar *_vt,                    \
    const size_t n_elem,            \
    const scalar _val)              \
{                                   \
    OUTER_LOOP {                    \
        _vt[i] op##= _val;          \
    }                               \
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

    
/** Compute basic math operations on vector elements give constant.
 *
 * @param _vt    - Pointer to input data.
 * @param n_elem - Number of elemets.
 * @param _val   - Constant value.
 */
extern void vi_add_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_sub_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_mul_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_div_s (scalar *_vt, const size_t n_elem, const scalar _val);


/** Compute basic matrix/matrix operations with added constant.
 *
 * \param mat       - Pointer the first matrix data.
 * \param mtb       - Pointer the second matrix data.
 * \param n_rows    - Number of rows of each matrix.
 * \param n_cols    - Numnber of columns in each matrix.
 * \param val       - Constant value.
 * \param buffer    - Pointer to output buffer.
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


/** Add two vectors element-wise.
 *
 * @param _vx    - Vector of size n_elem.
 * @param _vy    - Vector of size n_elem.
 * @param n_elem - Number of elements in each vector.
 * @param sum    - Output buffer of size n_elem.
 */
extern void
v_add (
    const scalar *restrict _vx,
    const scalar *restrict _vy,
    const size_t n_elem,
    scalar *sum);


/** Add first vector element-wise to second one.
 *
 * @param _vx    - Vector of size n_elem.
 * @param _vy    - Vector of size n_elem.
 * @param n_elem - Number of elements in each vector.
 */
extern void
vi_add (
    const scalar *restrict _vx,
    scalar *restrict _vy,
    const size_t n_elem);


/** Vectorized e function.
 */
extern void
v_exp (
    const scalar *restrict _vx,
    const size_t n_elem,
    scalar *_exps);


/** Vectorized e function inplace.
 */
extern void
vi_exp (
    scalar *restrict _vx,
    const size_t n_elem);


/** Vectorized logarithm.
 */
extern void
v_log (
    const scalar *restrict _vx,
    const size_t n_elem,
    scalar *_logs);


/** Vectorized logarithm inplace.
 */
extern void
vi_log (
    scalar *restrict _vx,
    const size_t n_elem);

/** Logarithm of the sum of the exponential of the vector elements.
 *
 * \param vctr   Pointer to input data.
 * \param n_elem Number of vector elements.
 */
extern scalar
v_lse (
    const scalar *restrict vctr,
    const size_t n_elem);


extern scalar
vs_lse_centroid (
    const scalar *restrict vt,
    const size_t v_stride,
    const scalar *restrict weights,
    const size_t w_stride,
    const size_t n_elem);

#define v_lse_centroid(vt, weights, n_elem) \
    vs_lse_centroid (vt, 1, weights, 1, n_elem)

/** Compute maximum element of vector.
 *
 * @param _vt
 * @param n_elem
 */
extern scalar
v_max (
    const scalar *restrict _vt,
    const size_t n_elem);


/** Compute the sum of the vector elements.
 *
 * @param _vt - Pointer to input data.
 * @param n_elem - Number of elements.
 */
extern scalar
v_sum (
    const scalar *restrict _vt,
    const size_t n_elem);

/* === Strided vector interface ==== */

/** Compute the sum of the vector elements given a stride.
 *
 * @param _vt - Pointer to input data.
 * @param n_elem - Number of elements.
 * @param stride - Stride.
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

/** Compute centroid along the rows of _mt.
 *
 * @param mtrx      Pointer to matrix data. 
 * @param wght      Pointer to weight data.
 * @param n_rows    Number of matrix rows.
 * @param n_cols    Number of matrix columns.
 * @param centroid  Output buffer.
 */
extern void
m_lse_centroid_rows (
        const scalar *restrict mtrx,
        const scalar *restrict wght,
        const size_t n_rows,
        const size_t n_cols,
        scalar *centroid);


/** Compute maximum value.
 *
 * @param _mat
 * @param _n_rows
 * @param _n_cols
 */
extern scalar
m_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols);


/** Compute maximum along matrix columns.
 *
 * @param _mat
 * @param _n_rows
 * @param _n_cols
 * @param _col_max
 */
extern void
m_col_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols,
    scalar *restrict _col_max);


/** Compute maximum along rows.
 *
 * @param _mat
 * @param _n_rows
 * @param _n_cols
 * @paran _row_max
 */
extern void
m_row_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols,
    scalar *restrict _row_max);


/** Compute maximum along columns.
 *
 * @param _mat
 * @param _n_rows
 * @param _n_cols
 * @paran _row_max
 */
extern void
m_col_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols,
    scalar *restrict _col_max);


/** Compute logarithm of matrix elements.
 *
 * \param mat    - Pointer to matrix elements.
 * \param n_elem - Number of matrix elements.
 * \param out    - Pointer to output buffer.
 */
#define m_log(mat, n_elem, out) v_log ((mat), (n_elem), (out))


/** Compute logarithm of matrix elements inplace
 *
 * \param mat    - Pointer to matrix elements.
 * \param n_elem - Number of matrix elements.
 */
#define mi_log(mat, n_elem) vi_log ((mat), (n_elem))


/** Compute vector/matrix product in log domain.
 *
 * Compute the product of a vector and a square matrix with the same
 * number of elements in each row and column. Computation is performed
 * in log domain by means of LSE.
 *
 * @param _vt    - Pointer to vector elements.
 * @param _mt    - Pointer to matrix elements.
 * @param n_elem - Number of elements in _vt. 
 * @param _cs    - Computation buffer of lenght n_elem.
 * @param _mb    _ Computation buffer of lenght n_elem^2.
 * @param prod   - Output buffer of lenght n_elem.
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
 * @param _mt    - Pointer to matrix elements.
 * @param _vt    - Pointer to vector elements.
 * @param n_elem - Number of elements in _vt. 
 * @param _cs    - Computation buffer of lenght n_elem.
 * @param _mb    _ Computation buffer of lenght n_elem^2.
 * @param prod   - Output buffer of lenght n_elem.
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
 * @param _buffer - Input buffer;
 * @param _n_elem - Number of elements in buffer.
 * @param _stride - Stride in elements.
 */
extern scalar
_strided_max (
    const scalar *restrict _buffer,
    const size_t _n_elem,
    const size_t _stride);

#endif  /* VMATH_H */
