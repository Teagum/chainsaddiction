#include "test_vmath.h"

unsigned short _N_ERRORS = 0;


bool
test_strided_max (void)
{
    const size_t stride = 4;
    const size_t n_mat_elem = stride * stride;
    const scalar max_val = 2.0L;
    scalar mat[n_mat_elem];

    v_rnd (n_mat_elem, mat);
    for (size_t i = 0; i < stride; i++)
    {
        mat[i*stride+i] = max_val;
    }
    return ASSERT_EQUAL (_strided_max (mat, n_mat_elem, stride), max_val);
}


bool
test_v_lse (void)
{
    const size_t n_elem = 100;
    scalar vals[n_elem];
    scalar lvals[n_elem];
    scalar lsum_vals = 0;
    scalar lse = 0;

    v_rnd (n_elem, vals);
    v_log (vals, n_elem, lvals);
    for (size_t i = 0; i < n_elem; i++) { lsum_vals += vals[i]; }
    lsum_vals = logl (lsum_vals);
    lse = v_lse (lvals, n_elem);
    return ASSERT_EQUAL (lsum_vals, lse);
}


bool
test_v_lse_centroid (void)
{
    const size_t n_elem = 3;
    scalar vals[n_elem] = {2, 3, 4};
    scalar lvals[n_elem];
    scalar weights[n_elem] = {5, 4, 6.0};
    scalar wsum = 0.0L;
    scalar sum = 0.0L;
    scalar cent = 0.L;
    scalar lse_cent = 0.0L;

    //v_rnd (n_elem, weights);
    //v_rnd (n_elem, vals);
    v_log (vals, n_elem, lvals);
    for (size_t i = 0; i < n_elem; i++) {
        sum += vals[i];
        wsum += vals[i] * weights[i];
    }
    cent = logl (wsum / sum);
    lse_cent = v_lse_centroid (lvals, weights, n_elem);
    return ASSERT_EQUAL (cent, lse_cent);
}


bool
test_v_max (void)
{
    scalar max = 2.0L;
    scalar vals[N];
    v_rnd (N, vals);
    vals[N/2] = max;
    if (ASSERT_EQUAL (v_max (vals, N), max))
    {
       return true;
    }
   return false;
}


bool
test_vs_sum (void)
{
    const size_t n_elem = 2;
    scalar vals[n_elem];
    size_t stride = 1; //rnd_int (0, n_elem);
    scalar expected = 0;
    scalar res = 0;

    v_rnd (n_elem, vals);
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        //printf ("[%3zu] %Lf\n", i, vals[i]);
        expected += vals[i];
    }
    res = vs_sum (vals, n_elem, stride);

    //printf ("RES: %Lf\tEXPECTED: %LF\tStride: %zu\n", res, expected, stride);
    return ASSERT_EQUAL (res, expected);
}


bool
test_m_max (void)
{
    scalar max = 2.0L;
    scalar vals[N];
    v_rnd (N, vals);
    vals[N/2] = max;
    if (ASSERT_EQUAL (m_max (vals, 2, N/2), max))
    {
       return true;
    }
   return false;
}


bool
test_m_row_max (void)
{
    const size_t rows = 5;
    const size_t cols = 5;
    const size_t n = rows * cols;
    scalar vals[n];
    int max_val[rows];
    scalar res_max[rows];

    v_rnd_int (10, 100, n, max_val);
    v_rnd (n, vals);

    for (size_t i = 0; i < rows; i++)
    {
        vals[i*rows+i] = (scalar) max_val[i];
    }

    m_row_max (vals, rows, cols, res_max);
    for (size_t i = 0; i < rows; i++)
    {
        if (!ASSERT_EQUAL (res_max[i], max_val[i]))
        {
            return false;
        }
    }
    return true;
}


bool
test_m_col_max (void)
{
    const size_t rows = 100;
    const size_t cols = 255;
    const size_t n = rows * cols;
    scalar vals[n];
    int max_val[cols];
    int max_row_idx[cols];
    scalar res_max[cols];

    v_rnd_int (0, rows, cols, max_row_idx);
    v_rnd_int (10, 100, n, max_val);
    v_rnd (n, vals);

    for (size_t i = 0; i < cols; i++)
    {
        size_t idx = max_row_idx[i] * cols + i;
        // printf ("MRI: %d\tIDX: %zu -> %d\n", max_row_idx[i], idx, max_val[i]);
        vals[idx] = (scalar) max_val[i];
    }

    /*
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf ("[%3zu] %7.4Lf  ", i*cols+j, vals[i*cols+j]);
        }
        printf("\n");
    }
    */

    m_col_max (vals, rows, cols, res_max);
    for (size_t i = 0; i < cols; i++)
    {
        // printf ("[%3zu] %Lf\n", i, res_max[i]);
        if (!ASSERT_EQUAL (res_max[i], max_val[i]))
        {
            return false;
        }
    }
    return true;
}

bool
test_log_vmp (void)
{
    const size_t n = 3;
    scalar vt[n] = {1, 2, 3};
    scalar mt[n*n] = {1, 2, 3, 2, 3, 1, 3, 2, 1};
    scalar b1[n];
    scalar b2[n*n];
    scalar res[n];

    v_rnd (n, vt);
    v_rnd (n*n, mt);
    vi_log (vt, n);
    vi_log (mt, n*n);

    log_vmp (vt, mt, n, b1, b2, res);
    return false;
}

bool
test_log_mvp (void)
{
    const size_t n = 3;
    scalar vt[n] = {1, 2, 3};
    scalar mt[n*n] = {1, 2, 3, 2, 3, 1, 3, 2, 1};
    scalar b1[n];
    scalar b2[n*n];
    scalar res[n];

    v_rnd (n, vt);
    v_rnd (n*n, mt);
    vi_log (vt, n);
    vi_log (mt, n*n);

    log_mvp (mt, vt, n, b1, b2, res);
    return false;
}
