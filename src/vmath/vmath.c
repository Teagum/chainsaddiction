#include "vmath.h"


/*
 * Low-level reductions
 */

inline scalar
v_acc_sum (size_t n_elem, size_t stride, scalar (*op) (scalar), const scalar *restrict vtx)
{
    scalar sum = 0.0L;
    if (n_elem == 0 || stride == 0)
    {
        errno = EDOM;
        return 0.0L;
    }

    if (op == NULL)
    {
        acc_sum (n_elem, stride, vtx, &sum);
    }
    else
    {
        acc_sum_op(n_elem, stride, op, vtx, &sum);
    }
    return sum;
}


inline scalar
v_acc_prod (size_t n_elem, size_t stride, scalar (*op) (scalar), const scalar *restrict vtx)
{
    scalar prod = 1.0L;
    if (n_elem == 0 || stride == 0)
    {
        errno = EDOM;
        return 0.0L;
    }

    if (op == NULL)
    {
        acc_prod(n_elem, stride, vtx, &prod);
    }
    else
    {
        acc_prod_op(n_elem, stride, op, vtx, &prod);
    }
    return prod;
}


/*
 * High-level reductions
 */
inline scalar
v_sum (size_t n_elem, const scalar *restrict vtx)
{
    return v_acc_sum (n_elem, 1, NULL, vtx);
}


inline scalar
v_sumlog (size_t n_elem, const scalar *restrict vtx)
{
    return v_acc_sum (n_elem, 1, logl, vtx);
}


inline scalar
v_sumexp (size_t n_elem, const scalar *restrict vtx)
{
    return v_acc_sum (n_elem, 1, expl, vtx);
}


inline scalar
v_lse (size_t n_elem, const scalar *restrict vtx)
{
    const scalar max_val = v_max (n_elem, vtx);
    scalar sum_exp = 0;
    for (size_t i = 0; i < n_elem; i++, vtx++)
    {
        sum_exp += expl (*vtx - max_val);
    }
    return logl (sum_exp) + max_val;
}


inline scalar
v_max (size_t n_elem, const scalar *restrict vtx)
{
    const scalar *max_ptr = vtx;
    while (--n_elem) {
        max_ptr = *++vtx >= *max_ptr ? vtx : max_ptr;
    }
    return *max_ptr;
}


inline scalar
v_min (size_t n_elem, const scalar *restrict vtx)
{
    const scalar *min_ptr = vtx;
    while (--n_elem) {
        min_ptr = *++vtx <= *min_ptr ? vtx : min_ptr;
    }
    return *min_ptr;
}


inline size_t
v_argmax (size_t n_elem, const scalar *restrict vtx)
{
    size_t arg = n_elem;
    size_t cnt = n_elem;
    const long double *max_ptr = vtx;
    while (--cnt) {
        if (*++vtx >= *max_ptr)
        {
            max_ptr = vtx;
            arg = cnt;
        }
    }
    return n_elem - arg;
}


inline size_t
v_argmin (size_t n_elem, const scalar *restrict vtx)
{
    size_t arg = n_elem;
    size_t cnt = n_elem;
    const long double *max_ptr = vtx;
    while (--cnt) {
        if (*++vtx <= *max_ptr)
        {
            max_ptr = vtx;
            arg = cnt;
        }
    }
    return n_elem - arg;
}


/*
 * Vectorized transforms
 */

def_v_op (exp, expl)
def_v_op (log, logl)
def_v_op (logr1, logr1)


inline void
v_softmax (size_t n_elem, const scalar *restrict vtx, scalar *restrict out)
{
    scalar total = 0.0L;
    scalar *iter = out;

    for (size_t i = 0; i < n_elem; i++)
    {
        *iter = expl (*vtx);
        total += *iter;
        iter++;
        vtx++;
    }
    iter = NULL;

    for (size_t i = 0; i < n_elem; i++)
    {
        *out /= total;
        out++;
    }
}


void
v_log_normalize (
    size_t n_elem,
    const scalar *restrict lvtx,
    scalar *restrict out)
{
    const scalar lsum = v_lse (n_elem, lvtx);
    for (size_t i = 0; i < n_elem; i++)
    {
        *out++ = *lvtx++ - lsum;
    }
}


/*
 * Vectorized inplace transforms
 */

def_vi_op(exp, expl)
def_vi_op(log, logl)
def_vi_op(logr1, logr1)


inline void
vi_softmax (size_t n_elem, scalar *vtx)
{
    scalar total = 0.0L;
    scalar *iter = vtx;

    for (size_t i = 0; i < n_elem; i++)
    {
        *iter = expl (*iter);
        total += *iter;
        iter++;
    }
    iter = NULL;
    vsi_div (n_elem, total, vtx);
}


void
vi_log_normalize (
    size_t n_elem,
    scalar *restrict vtx)
{
    scalar lsum = v_lse (n_elem, vtx);
    for (size_t i = 0; i < n_elem; i++)
    {
        *vtx++ -= lsum;
    }
}


/*
 * Basic vector/scalar arithmetic
 */
def_vs_op(add, +)
def_vs_op(sub, -)
def_vs_op(mul, *)
def_vs_op(div, /)


/*
 * Basic vector/scalar inplace arithmetic
 */
def_vsi_op(add, +)
def_vsi_op(sub, -)
def_vsi_op(mul, *)
def_vsi_op(div, /)


/*
 * Basic vector/vector arithmetic
 */
def_vv_op(add, +)
def_vv_op(sub, -)
def_vv_op(mul, *)
def_vv_op(div, /)


/*
 * Basic vector/vector inplace arithmetic
 */
def_vvi_op(add, +)
def_vvi_op(sub, -)
def_vvi_op(mul, *)
def_vvi_op(div, /)

def_mm_op_s_func(add, +)
def_mm_op_s_func(sub, -)
def_mm_op_s_func(mul, *)
def_mm_op_s_func(div, /)







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
    scalar max_val = v_max (n_elem, vt);

    for (size_t i = 0; i < n_elem; i++, vt+=v_stride, weights+=w_stride)
    {
        scalar _buff = expl (*vt - max_val);
        sum_exp += _buff;
        sum_exp_w += _buff * (*weights);
    }
    return logl (sum_exp_w/sum_exp);
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
    return v_max (_n_rows*_n_cols, _mt);
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
        row_max[i] = v_max (n_cols, mtx);
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
    return VM_SUCCESS;
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
    return VM_SUCCESS;
}


extern void
vm_add (
    const size_t rows,
    const size_t cols,
    const scalar *const restrict vtx,
    const scalar *restrict mtx,
    scalar *restrict out)
{
    for (size_t i = 0; i < rows; i++)
    {
        vv_add (cols, vtx, mtx, out);
        mtx+=cols;
        out+=cols;
    }
}


extern void
vmi_add (
    const size_t rows,
    const size_t cols,
    const scalar *const restrict vtx,
    scalar *restrict mtx)
{
    for (size_t i = 0; i < rows; i++)
    {
        vvi_add (cols, vtx, mtx);
        mtx+=cols;
    }
}

extern void
vm_multiply (const size_t rows, const size_t cols, const scalar *const vtx,
             const scalar *const mtx, scalar *restrict prod)
{
    const scalar *vtx_data = NULL;
    const scalar *mtx_data = NULL;
          scalar *out_data = prod;

    for (size_t i = 0; i < cols; i++)
    {
        vtx_data  = vtx;
        mtx_data  = mtx + i;
        *out_data = 0.0L;
        for (size_t j = 0; j < rows; j++)
        {
            *out_data = fmal (*vtx_data, *mtx_data, *out_data);
            vtx_data++;
            mtx_data+=cols;
        }
        out_data++;
    }
}


inline void
vm_multiply_log (
    const size_t rows,
    const size_t cols,
    const scalar *const vtx,
    const scalar *const mtx,
          scalar *const acc,
          scalar *restrict prod)
{
    const scalar *vt_data  = NULL;
    const scalar *mt_data  = NULL;
          scalar *acc_data = NULL;
          scalar row_max   = 0.0L;

    for (size_t i = 0; i < cols; i++) {
        vt_data  = vtx;
        mt_data  = mtx + i;
        acc_data = acc;
        row_max  = -INFINITY;
        *prod    = 0.0L;

        for (size_t j = 0; j < rows; j++) {
            *acc_data = *mt_data + *vt_data++;
            row_max = fmax (*acc_data, row_max);
            acc_data++;
            mt_data+=cols;
        }

        acc_data = acc;
        for (size_t j = 0; j < rows; j++) {
            *prod += expl (*acc_data++ - row_max);
        }

        *prod = logl (*prod) + row_max;
        prod++;
    }
}


/*
 * ============================================================================
 * Matrix * vector interface
 * ============================================================================
 */

extern void
mv_multiply (const size_t rows, const size_t cols, const scalar *const mtx,
             const scalar *const vtx, scalar *restrict out)
{
    const scalar *v_data = NULL;
    const scalar *m_data = mtx;

    for (size_t i = 0; i < rows; i++)
    {
        *out = 0.0L;
        v_data = vtx;
        for (size_t j = 0; j < cols; j++)
        {
            *out = fmal (*m_data++, *v_data++, *out);
        }
        out++;
    }
}


extern void
mv_multiply_log (
    const size_t rows,
    const size_t cols,
    const scalar *const mtx,
    const scalar *const vtx,
          scalar *const acc,
          scalar *restrict prod)
{
    const scalar *vt_data  = vtx;
    const scalar *mt_data  = mtx;
          scalar *acc_data = acc;
          scalar row_max   = -INFINITY;

    for (size_t i = 0; i < rows; i++)
    {
        *prod = 0.0L;
        for (size_t j = 0; j < cols; j++)
        {
            *acc_data = *mt_data++ + *vt_data++;
            row_max = fmax (*acc_data, row_max);
            acc_data++;
        }

        acc_data = acc;
        for (size_t j = 0; j < cols; j++)
        {
            *prod += expl (*acc_data++ - row_max);
        }
        *prod = logl (*prod) + row_max;

        prod++;
        vt_data = vtx;
        acc_data = acc;
        row_max = -INFINITY;
    }
}

/*
 * ============================================================================
 * Matrix/matrix interface
 * ============================================================================
 */

void mm_multiply (const size_t xr, const size_t rc, const size_t yc,
                  const scalar *mtx, const scalar *mty, scalar *out)
{
    const scalar *x_row_ptr = NULL;
    const scalar *y_col_ptr = NULL;

    for (size_t i = 0; i < xr; i++)
    {
        for (size_t j = 0; j < yc; j++)
        {
            x_row_ptr = mtx;
            y_col_ptr = mty+j;
            for (size_t k = 0; k < rc; k++)
            {
                *out += *x_row_ptr * *y_col_ptr;
                x_row_ptr++;
                y_col_ptr+=yc;
            }
            out++;
        }
        mtx+=rc;
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


inline scalar
logr1 (scalar val)
{
    if (isnormal (val))
    {
        return logl (val);
    }
    else
    {
        return 1.0L;
    }
}


inline void
mi_row_apply (size_t rows, size_t cols, void (*row_op) (size_t, scalar *), scalar *mtx)
{
    for (size_t i = 0; i < rows; i++)
    {
        row_op (cols, mtx);
        mtx+=cols;
    }
}
