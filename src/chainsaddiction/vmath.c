#include "vmath.h"

inline void
v_add (
    const scalar *restrict _vx,
    const scalar *restrict _vy,
    const size_t n_elem,
    scalar *sum)
{
    OUTER_LOOP {
        sum[i] = _vx[i] + _vy[i];
    }
}

inline void
vi_add (
    const scalar *restrict _vx,
    scalar *_vy,
    const size_t n_elem)
{
    OUTER_LOOP {
        _vy[i] += _vx[i];
    }
}

inline void
v_exp (
    const scalar *restrict _vx,
    const size_t n_elem,
    scalar *_exps)
{
    OUTER_LOOP {
        _exps[i] = expl (_vx[i]);
    }
}

inline void
vi_exp (
    scalar *restrict _vx,
    const size_t n_elem)
{
    OUTER_LOOP {
        _vx[i] = expl (_vx[i]);
    }
}

inline void
v_log (
    const scalar *restrict _vx,
    const size_t n_elem,
    scalar *_logs)
{
    OUTER_LOOP {
        _logs[i] = logl (_vx[i]);
    }
}


inline void
vi_log (
    scalar *restrict _vx,
    const size_t n_elem)
{
    OUTER_LOOP {
        _vx[i] = logl (_vx[i]);
    }
}


inline scalar
v_lse (
    const scalar *restrict _vx,
    const size_t n_elem)
{
    const scalar max_val = v_max (_vx, n_elem);
    scalar sum_exp = 0;
    for (size_t i = 0; i < n_elem; i++, _vx++)
    {
        sum_exp += expl (*_vx - max_val);
    }
    return logl (sum_exp) + max_val;
}


inline scalar
v_max (
    const scalar *restrict _vt,
    const size_t n_elem)
{
    scalar _max = *_vt++;
    for (size_t i = 1; i < n_elem; i++, _vt++)
    {
        _max = fmaxl (*_vt, _max);
    }
    return _max;
}


inline scalar
v_sum (
    const scalar *restric _vt,
    const size_t n_elem)
{
    return vs_sum (_vt, n_elem, 1);
}


inline scalar
vs_sum (
    const scalar *restrict _vt,
    const size_t n_elem,
    const size_t stride)
{
    scalar sum = 0;
    const scalar *end_iter = _vt + n_elem;
    while ((_vt+=stride) < end_iter)
    {
        sum += *_vt;
    }
    return sum;
}


inline void
m_log_centroid (
        const scalar *restrict _mt,
        const scalar *restrict _weights,
        const size_t _n_rows,
        const size_t _n_cols,
        scalar _centroid)
{
    scalar *max_per_col = _alloc_block (_n_cols);
    scalar *sum_per_col = _alloc_block_fill (_n_cols, 0.0L);

    for (size_t i = 0; i < _n_cols; i++)
    {
        max_per_col[i] = _mt[i];
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
    const scalar *restrict _mt,
    const size_t _n_rows,
    const size_t _n_cols,
    scalar *restrict _row_max)
{
    for (size_t i = 0; i < _n_cols; i++, _mt+=_n_cols)
    {
        _row_max[i] = v_max (_mt, _n_cols);
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
    const scalar *restrict _vt,
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
            _mb[idx] = _mt[idx] + _vt[j];
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
    const scalar *restrict _vt,
    const size_t n_elem,
    scalar *_cs,
    scalar *_mb,
    scalar *_prod)
{
    OUTER_LOOP {
        _cs[i] = -INFINITY;
        INNER_LOOP {
            size_t idx = i * n_elem + j;
            _mb[idx] = _mt[idx] + _vt[j];
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
    const scalar *restrict _buffer,
    const size_t _n_elem,
    const size_t _stride)
{
    const scalar *end_iter = _buffer + _n_elem;
    scalar c_max = *_buffer;
    while ((_buffer+=_stride) < end_iter)
    {
        c_max = fmaxl (*_buffer, c_max);
    }
    return c_max;
}
