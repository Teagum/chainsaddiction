#include "vmath.h"

def_vi_s_func(add, +)
def_vi_s_func(sub, -)
def_vi_s_func(mul, *)
def_vi_s_func(div, /)

def_mm_op_s_func(add, +)
def_mm_op_s_func(sub, -)
def_mm_op_s_func(mul, *)
def_mm_op_s_func(div, /)


inline void
v_add (
    scalar *const vx,
    scalar *const vy,
    const size_t n,
    scalar *out)
{
    FOR_EACH(i, n) {
        out[i] = vx[i] + vy[i];
    }
}

inline void
vi_add (
    const scalar *restrict vx,
    scalar *vy,
    const size_t n_elem)
{
    OUTER_LOOP {
        vy[i] += vx[i];
    }
}

inline void
v_exp (
    const scalar *restrict vx,
    const size_t n_elem,
    scalar *_exps)
{
    OUTER_LOOP {
        _exps[i] = expl (vx[i]);
    }
}

inline void
vi_exp (
    scalar *restrict vx,
    const size_t n_elem)
{
    OUTER_LOOP {
        vx[i] = expl (vx[i]);
    }
}

inline void
v_log (
    const scalar *restrict vx,
    const size_t n_elem,
    scalar *_logs)
{
    OUTER_LOOP {
        _logs[i] = logl (vx[i]);
    }
}


inline void
vi_log (
    scalar *restrict vx,
    const size_t n_elem)
{
    OUTER_LOOP {
        vx[i] = logl (vx[i]);
    }
}


inline scalar
v_lse (
    const scalar *restrict vctr,
    const size_t n_elem)
{
    const scalar max_val = v_max (vctr, n_elem);
    scalar sum_exp = 0;
    for (size_t i = 0; i < n_elem; i++, vctr++)
    {
        sum_exp += expl (*vctr - max_val);
    }
    return logl (sum_exp) + max_val;
}


inline scalar
vs_lse_centroid (
    const scalar *restrict vt,
    const size_t v_stride,
    const scalar *restrict weights,
    const size_t w_stride,
    const size_t n_elem)
{
    scalar sum_exp =  0.0L;
    scalar sum_exp_w = 0.0L;
    scalar max_val = v_max (vt, n_elem);

    for (size_t i = 0; i < n_elem; i++, vt+=v_stride, weights+=w_stride)
    {
        scalar _buff = expl (*vt - max_val);
        sum_exp += _buff;
        sum_exp_w += _buff * (*weights);
    }
    return logl (sum_exp_w/sum_exp);
}


inline scalar
v_max (
    const scalar *restrict vt,
    const size_t n_elem)
{
    scalar _max = *vt++;
    for (size_t i = 1; i < n_elem; i++, vt++)
    {
        _max = fmaxl (*vt, _max);
    }
    return _max;
}


inline void
vi_softmax (scalar *buffer, size_t n_elem)
{
    scalar total = 0.0L;
    for (size_t i = 0; i < n_elem; i++)
    {
        buffer[i] = expl (buffer[i]);
        total += buffer[i];
    }
    for (size_t i = 0; i < n_elem; i++)
    {
        buffer[i] /= total;
    }
}


inline scalar
v_sum (
    const scalar *restrict vt,
    const size_t n_elem)
{
    return vs_sum (vt, n_elem, 1);
}


inline scalar
vs_sum (
    const scalar *restrict vt,
    const size_t n_elem,
    const size_t stride)
{
    scalar sum = 0;
    for (size_t i = 0; i < n_elem; i+=stride, vt+=stride)
    {
        sum += *vt;
    }
    return sum;
}

/*!*********************************************
 * Matrix interface
 */

inline int
m_lse_centroid_rows (
    const scalar *restrict mtrx,
    const scalar *restrict wght,
    const size_t n_rows,
    const size_t n_cols,
    scalar *centroid)
{
    int err = 1;
    scalar *row_sum = VA_SCALAR_ZEROS (n_cols);
    scalar *w_row_sum = VA_SCALAR_ZEROS (n_cols);
    scalar *row_max = VA_SCALAR_ZEROS (n_cols);

    if (row_sum == NULL || w_row_sum == NULL || row_max == NULL)
    {
        fputs ("Could not allocate buffer in `m_lse_centroid_rows'.", stderr);
        err = 1;
    }
    else
    {
        m_row_max (mtrx, n_rows, n_cols, row_max);
        for (size_t i = 0; i < n_rows*n_cols; i++)
        {
            size_t idx = i % n_cols;
            scalar exp_val = expl (mtrx[i] - row_max[idx]);
            row_sum[idx] += exp_val;
            w_row_sum[idx] += exp_val * wght[i/n_cols];
        }

        for (size_t i = 0; i < n_cols; i++)
        {
            centroid[i] = logl (w_row_sum[i] / row_sum[i]);
        }
        err = 0;
    }

    FREE (row_sum);
    FREE (w_row_sum);
    FREE (row_max);
    return err;
}


inline scalar
m_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols)
{
    return v_max (_mt, _n_rows*_n_cols);
}


inline void
m_row_max (
    const scalar *restrict mt,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict row_max)
{
    for (size_t i = 0; i < n_cols; i++, mt+=n_cols)
    {
        row_max[i] = v_max (mt, n_cols);
    }
}


inline void
m_col_max (
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols,
    scalar *restrict _col_max)
{
    for (size_t i = 0; i < _n_cols; i++, _mt++)
    {
        _col_max[i] = _strided_max (_mt, _n_rows*_n_cols, _n_cols);
    }
}

inline void
log_vmp (
    const scalar *restrict vt,
    const scalar *restrict _mt,
    const size_t n_elem,
    scalar *_cs,
    scalar *_mb,
    scalar *_prod)
{
    OUTER_LOOP {
        _cs[i] = -INFINITY;
        INNER_LOOP {
            size_t idx = j * n_elem + i;
            _mb[idx] = _mt[idx] + vt[j];
            _cs[i] = fmax (_mb[idx], _cs[i]);
        }
    }

    OUTER_LOOP {
        _prod[i] = 0.0L;
        INNER_LOOP {
            size_t idx = j * n_elem + i;
            _prod[i] += expl (_mb[idx]-_cs[i]);
        }
        _prod[i] = logl (_prod[i]) + _cs[i];
    }
}

inline void
log_mvp (
    const scalar *restrict _mt,
    const scalar *restrict vt,
    const size_t n_elem,
    scalar *_cs,
    scalar *_mb,
    scalar *_prod)
{
    OUTER_LOOP {
        _cs[i] = -INFINITY;
        INNER_LOOP {
            size_t idx = i * n_elem + j;
            _mb[idx] = _mt[idx] + vt[j];
            _cs[i] = fmax(_mb[idx], _cs[i]);
        }
    }

    OUTER_LOOP {
        _prod[i] = 0.0L;
        INNER_LOOP {
            size_t idx = i * n_elem + j;
            _prod[i] += expl (_mb[idx]-_cs[i]);
        }
        _prod[i] = logl (_prod[i]) + _cs[i];
    }
}


/*
 * Private API
 */
scalar
_strided_max (
    const scalar *restrict buffer,
    const size_t n_elem,
    const size_t stride)
{
    scalar c_max = 0;
    for (size_t i = 0; i < n_elem; i+=stride, buffer+=stride)
    {
        c_max = fmaxl (*buffer, c_max);
    }
    return c_max;
}
