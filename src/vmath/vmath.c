#include "vmath.h"


/*
 * Vector/scalar operators
 */
def_vs_op(add, +)
def_vs_op(sub, -)
def_vs_op(mul, *)
def_vs_op(div, /)

/*
 * Vector/scalar inplace operators
 */
def_vsi_op(add, +)
def_vsi_op(sub, -)
def_vsi_op(mul, *)
def_vsi_op(div, /)

/*
 * Vector/vector arithmetic
 */
def_vv_op(add, +)
def_vv_op(sub, -)
def_vv_op(mul, *)
def_vv_op(div, /)

/*
 * Vector/vector inplace arithmetic
 */
def_vvi_op(add, +)
def_vvi_op(sub, -)
def_vvi_op(mul, *)
def_vvi_op(div, /)

def_mm_op_s_func(add, +)
def_mm_op_s_func(sub, -)
def_mm_op_s_func(mul, *)
def_mm_op_s_func(div, /)


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

inline void
v_logr1 (
    scalar *restrict vct,
    const size_t n_elem,
    scalar *restrict out)
{
    FOR_EACH (i, n_elem)
    {
        *out = logr1 (*vct);
        out++;
        vct++;
    }
}


inline void
vi_logr1 (
    scalar *restrict vct,
    const size_t n_elem)
{
    FOR_EACH (i, n_elem)
    {
        *vct = logr1 (*vct);
        vct++;
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


extern size_t
v_argmax (const size_t n_elem, const scalar *restrict vec)
{
    size_t arg = 0;
    scalar max = *vec;

    for (size_t i = 1; i < n_elem; i++)
    {
        if (vec[i] > max)
        {
            arg = i;
            max = vec[i];
        }
    }
    return arg;
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
m_log_centroid_cols (
    const scalar *restrict mtx,
    const scalar *restrict wgt,
    const size_t n_rows,
    const size_t n_cols,
    scalar *centroid_in_col)
{
    int err = 1;
    scalar *max_per_col = VA_SCALAR_ZEROS (n_cols);
    scalar *sum_per_col = VA_SCALAR_ZEROS (n_cols);
    scalar *w_sum_per_col = VA_SCALAR_ZEROS (n_cols);

    if (sum_per_col == NULL || w_sum_per_col == NULL || max_per_col == NULL)
    {
        fputs ("Virtual memory exhausted in `m_lse_centroid_rows'.", stderr);
        err = 1;
    }
    else
    {
        m_col_max (mtx, n_rows, n_cols, max_per_col); /* BUG */
        for (size_t i = 0; i < n_rows*n_cols; i++)
        {
            size_t c = i % n_cols;
            scalar exp_val = expl (mtx[i] - max_per_col[c]);
            sum_per_col[c] += exp_val;
            w_sum_per_col[c] += exp_val * wgt[i/n_cols];
        }

        for (size_t i = 0; i < n_cols; i++)
        {
            /* ceck for division by zero */
            centroid_in_col[i] = logl (w_sum_per_col[i] / sum_per_col[i]);
        }
        err = 0;
    }

    FREE (sum_per_col);
    FREE (w_sum_per_col);
    FREE (max_per_col);
    return err;
}


inline void
m_row_argmax (
    const size_t rows,
    const size_t cols,
    const scalar *mtx,
    size_t *restrict argmax_row)
{
    for (size_t i = 0; i < rows; i++, mtx+=cols)
    {
        argmax_row[i] = v_argmax (cols, mtx);
    }
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
    const scalar *restrict mtx,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict row_max)
{
    for (size_t i = 0; i < n_rows; i++, mtx+=n_cols)
    {
        row_max[i] = v_max (mtx, n_cols);
    }
}


inline int
m_col_max (
    const scalar *restrict mtx,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict max_per_col)
{
    if (n_rows == 0 && n_cols == 0)
    {
        fprintf (stderr, "Maximum of zero sized buffer if not defined.\n");
        return VM_ERR_ZERO_SIZED_BUFFER;
    }
    size_t n_elem = n_rows * n_cols;
    for (size_t i = 0; i < n_cols; i++, mtx++)
    {
        max_per_col[i] = strided_max (n_elem--, n_cols, mtx);
    }
    return SUCCESS;
}


inline int
m_col_absmax (
    const scalar *restrict mtx,
    const size_t n_rows,
    const size_t n_cols,
    scalar *restrict max_per_col)
{
    if (n_rows == 0 && n_cols == 0)
    {
        fprintf (stderr, "Absolute maximum of zero sized buffer if not defined.\n");
        return VM_ERR_ZERO_SIZED_BUFFER;
    }
    size_t n_elem = n_rows * n_cols;
    for (size_t i = 0; i < n_cols; i++, mtx++)
    {
        max_per_col[i] = strided_absmax (n_elem--, n_cols, mtx);
    }
    return SUCCESS;
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
strided_max (
    const size_t n_elem,
    const size_t stride,
    const scalar *restrict buffer)
{
    const scalar *end = buffer + n_elem - 1;
    scalar c_max = *buffer;

    while ((buffer+=stride) <= end)
    {
        c_max = fmaxl (*buffer, c_max);
    }
    return c_max;
}


scalar
strided_absmax (
    const size_t n_elem,
    const size_t stride,
    const scalar *restrict buffer)
{
    const scalar *end = buffer + n_elem - 1;
    scalar c_max = *buffer;

    while ((buffer+=stride) <= end)
    {
        c_max = fmaxl (fabsl (*buffer), c_max);
    }
    return c_max;
}
