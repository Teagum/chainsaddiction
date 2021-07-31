#ifndef VMATH_H
#define VMATH_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "config.h"

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
#define ASSERT_ALLOC(buff) if (buff == NULL) {          \
    fputs ("Could not allocate buffer.\n", stderr);     \
    return 1;                                           \
}

#define SUCCESS 0

enum vmath_error_codes {
    VM_ERR_ZERO_SIZED_BUFFER = -1,
};

#define VA_SCALAR_EMPTY(n_elem) malloc ((n_elem) * sizeof (scalar));
#define VA_SCALAR_ZEROS(n_elem) calloc (n_elem, sizeof (scalar))
#define VA_INT_EMPTY(n_elem) malloc ((n_elem) * sizeof (int))
#define VA_INT_ZEROS(n_elem) calloc (n_elem, sizeof (int))
#define VA_SIZE_EMPTY(n_elem) malloc ((n_elem) * sizeof (size_t));
#define VA_SIZE_ZEROS(n_elem) calloc (n_elem, sizeof (size_t));


#define FREE(buff) do { \
    free (buff);        \
    buff = NULL;        \
} while (0)

#define FOR_EACH(idx, max) for (size_t (idx) = 0; (idx) < (max); (idx++))

#define OUTER_LOOP for (size_t i = 0; i < n_elem; i++)
#define INNER_LOOP for (size_t j = 0; j < n_elem; j++)

#define M_OUTER_LOOP for (size_t i = 0; i < _n_rows; i++)
#define M_INNER_LOOP for (size_t j = 0; j < _n_cols; j++)
#define ITER_MATRIX \
   M_OUTER_LOOP { \
      M_INNER_LOOP { \
#define END_ITER_MATRIX }}


#define NEWLINE fputc ('\n', stderr)
#define RED "\x1b[33m"
#define GREEN "\x1b[32m"
#define YELLOW "\x1b[34m"
#define CLEAR "\x1b[0m"


#define print_vector(n, vct) do {               \
    NEWLINE;                                    \
    fprintf (stderr, "%6c", ' ');               \
    for (size_t i = 0; i < n; i++) {            \
        fprintf (stderr, YELLOW "%6c[%2zu] " CLEAR, ' ', i);                  \
    }                                           \
    NEWLINE;                                    \
    fprintf (stderr, "%6c", ' ');\
    for (size_t i = 0; i < n; i++) {            \
        fprintf (stderr, "%10.5Lf ",  (scalar)vct[i]);  \
    }                                           \
    NEWLINE;                                    \
} while (0)


#define print_matrix(rows, cols, mtx) do {      \
    NEWLINE;\
    fprintf (stderr, "%6c", ' ');               \
    for (size_t i = 0; i < cols; i++) {            \
        fprintf (stderr, GREEN "%6c[%2zu] " CLEAR, ' ', i);                  \
    }                                           \
    NEWLINE;\
    for (size_t i = 0; i < rows; i++) {         \
        fprintf (stderr, GREEN "[%3zu] " CLEAR, i);                  \
        for (size_t j = 0; j < cols; j++) {     \
            fprintf (stderr, "%10.5Lf ", (scalar)mtx[i*cols+j]); \
        }                                       \
        NEWLINE;                                \
    }                                           \
} while (0)

#define logr1(val) isnormal (val) ? logl (val) : 1L

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
 * \param _vt       Pointer to input data.
 * \param n_elem    Number of elemets.
 * \param _val      Constant value.
 */
extern void vi_add_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_sub_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_mul_s (scalar *_vt, const size_t n_elem, const scalar _val);
extern void vi_div_s (scalar *_vt, const size_t n_elem, const scalar _val);


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


/** Add two vectors element-wise.
 *
 * \param[in]  vtx       Vector of size n_elem.
 * \param[in]  vty       Vector of size n_elem.
 * \param[in]  n_elem    Number of elements in each vector.
 * \param[out] out       Output buffer of size n_elem.
 */
extern void
v_add (
    const scalar *const vx,
    const scalar *const vy,
    const size_t n,
    scalar *out);


/** Add first vector element-wise to second one.
 *
 * \param _vx       Vector of size n_elem.
 * \param _vy       Vector of size n_elem.
 * \param n_elem    Number of elements in each vector.
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


/** Replace non-normal values with 1 in log domain.
 *
 * \param[in]  vct      Pointer to input vector.
 * \param[in]  n_elem   Number of elements in input vector.
 * \param[out] out      Pointer to output buffer.
 */
extern void
v_logr1 (
    scalar *restrict vct,
    const size_t n_elem,
    scalar *restrict out);


/** Replace non-normal values with 1 in log domain inplace.
 *
 * \param[in]  vct      Pointer to input vector.
 * \param[in]  n_elem   Number of elements in input vector.
 */
extern void
vi_logr1 (
    scalar *restrict vct,
    const size_t n_elem);


/** Logarithm of the sum of the exponential of the vector elements.
 *
 * \param vctr      Pointer to input data.
 * \param n_elem    Number of vector elements.
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
 * \param _vt
 * \param n_elem
 */
extern scalar
v_max (const scalar *restrict _vt, const size_t n_elem);


/** Compute softmax o `buffer' inplace.
 *
 * \param buffer    Pointer to object.
 * \param n_elem    Number of elements in object.
 */
extern void
vi_softmax (scalar *buffer, size_t n_elem);


/** Compute the sum of the vector elements.
 *
 * \param _vt       Pointer to input data.
 * \param n_elem    Number of elements.
 */
extern scalar
v_sum (
    const scalar *restrict _vt,
    const size_t n_elem);

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


/** Compute logarithm of matrix elements.
 *
 * \param mat       Pointer to matrix elements.
 * \param n_elem    Number of matrix elements.
 * \param out       Pointer to output buffer.
 */
#define m_log(mat, n_elem, out) v_log ((mat), (n_elem), (out))


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


#endif  /* VMATH_H */
