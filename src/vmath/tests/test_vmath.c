#include "unittest.h"
#include "test_vmath.h"


bool
test__strided_max (void)
{
    const size_t stride = 4;
    const size_t n_elem = stride * stride;
    const scalar max_val = 2.0L;
    scalar *mat = VA_SCALAR_EMPTY (n_elem);
    bool err = true;

    v_rnd (n_elem, mat);
    for (size_t i = 0; i < stride; i++)
    {
        assert (i*stride+i < n_elem);
        mat[i*stride+i] = max_val;
    }
    err = !ASSERT_EQUAL (_strided_max (mat, n_elem, stride), max_val);

    FREE (mat);
    return err;
}


bool
test__v_lse (void)
{
    const size_t n_elem = 100;
    scalar *vals = VA_SCALAR_EMPTY (n_elem);
    scalar *lvals = VA_SCALAR_EMPTY (n_elem);
    scalar lsum_vals = 0;
    scalar lse = 0;
    bool err = true;

    v_rnd (n_elem, vals);
    v_log (vals, n_elem, lvals);
    for (size_t i = 0; i < n_elem; i++) { lsum_vals += vals[i]; }
    lsum_vals = logl (lsum_vals);
    lse = v_lse (lvals, n_elem);

    err = !ASSERT_EQUAL (lsum_vals, lse);
    FREE (vals);
    FREE (lvals);
    return err;
}


bool
test__vs_lse_centroid (void)
{
    const size_t n_elem = 3;
    scalar vals[] = {2, 3, 4};
    scalar lvals[] = { 0.0, 0.0, 0.0 };
    scalar weights[] = {5, 4L, 6.0};
    scalar wsum = 0.0L;
    scalar sum = 0.0L;
    scalar cent = 0.L;
    scalar lse_cent = 0.0L;

    const size_t v_stride = 1;
    const size_t w_stride = 1;

    v_rnd (n_elem, weights);
    v_rnd (n_elem, vals);
    v_log (vals, n_elem, lvals);
    for (size_t i = 0; i < n_elem; i++) {
        sum += vals[i];
        wsum += vals[i] * weights[i];
    }
    cent = logl (wsum / sum);
    lse_cent = vs_lse_centroid (lvals, v_stride, weights, w_stride, n_elem);
    return !ASSERT_EQUAL (cent, lse_cent);
}


bool
test__v_max (void)
{
    bool err = true;
    scalar max = 2.0L;
    size_t n_elem = 100;
    scalar *vals = VA_SCALAR_EMPTY (n_elem);

    v_rnd (n_elem, vals);
    vals[n_elem/2] = max;
    err = !ASSERT_EQUAL (v_max (vals, n_elem), max);

    FREE (vals);
    return err;
}


bool
test__vs_sum (void)
{
    const size_t n_elem = 2;
    const size_t stride = 1; //rnd_int (0, n_elem);
    scalar *vals = VA_SCALAR_EMPTY (n_elem);
    scalar expected = 0;
    scalar vs_sum_res = 0;
    bool err = true;

    v_rnd (n_elem, vals);
    for (size_t i = 0; i < n_elem; i+=stride)
    {
        expected += vals[i];
    }
    vs_sum_res = vs_sum (vals, n_elem, stride);
    err = !ASSERT_EQUAL (vs_sum_res, expected);

    FREE (vals);
    return err;
}


bool
test__m_max (void)
{
    scalar max = 2L;
    scalar vals[N];

    v_rnd (N, vals);
    vals[N/2] = max;

    return !ASSERT_EQUAL (m_max (vals, 2, N/2), max);
}


bool
test__m_row_max (void)
{
    const size_t rows = 5;
    const size_t cols = 5;
    const size_t n_elem = rows * cols;
    int *max_val = VA_INT_EMPTY (rows);
    scalar *vals = VA_SCALAR_EMPTY (n_elem);
    scalar *res_max = VA_SCALAR_EMPTY (rows);
    bool err = true;

    v_rnd_int (10, 100, rows, max_val);
    v_rnd (n_elem, vals);

    for (size_t i = 0; i < rows; i++)
    {
        assert (i*rows+i < n_elem);
        vals[i*rows+i] = (scalar) max_val[i];
    }

    m_row_max (vals, rows, cols, res_max);
    for (size_t i = 0; i < rows; i++)
    {
        err &= ASSERT_EQUAL (res_max[i], max_val[i]);
    }
    FREE (max_val);
    FREE (vals);
    FREE (res_max);
    return !err;
}


bool
test__m_col_max (void)
{
    const size_t rows = 4;
    const size_t cols = 3;
    const size_t n_elem = rows * cols;
    scalar *vals = VA_SCALAR_EMPTY (n_elem);
    int *max_val = VA_INT_EMPTY (cols);
    int *max_row_idx = VA_INT_EMPTY (cols);
    scalar *res_max = VA_SCALAR_EMPTY (cols);
    bool err = true;

    v_rnd_int (0, rows, cols, max_row_idx);
    v_rnd_int (10, 100, cols, max_val);
    v_rnd (n_elem, vals);

    for (size_t i = 0; i < cols; i++)
    {
        size_t idx = max_row_idx[i] * cols + i;
        vals[idx] = (scalar) max_val[i];
    }

    m_col_max (vals, rows, cols, res_max);
    for (size_t i = 0; i < cols; i++)
    {
        // printf ("[%3zu] %Lf\n", i, res_max[i]);
        err &= ASSERT_EQUAL (res_max[i], max_val[i]);
    }
    FREE (vals);
    FREE (max_val);
    FREE (max_row_idx);
    FREE (res_max);
    return !err;
}


bool
test__log_vmp (void)
{
    const size_t n_elem = 3;
    scalar *vt = VA_SCALAR_EMPTY (n_elem);
    scalar *mt = VA_SCALAR_EMPTY (n_elem*n_elem);
    scalar *b1 = VA_SCALAR_EMPTY (n_elem);
    scalar *b2 = VA_SCALAR_ZEROS (n_elem*n_elem);
    scalar *res = VA_SCALAR_ZEROS (n_elem);

    v_rnd (n_elem, vt);
    v_rnd (n_elem*n_elem, mt);
    vi_log (vt, n_elem);
    vi_log (mt, n_elem*n_elem);

    log_vmp (vt, mt, n_elem, b1, b2, res);

    FREE (vt);
    FREE (mt);
    FREE (b1);
    FREE (b2);
    FREE (res);
    return false;
}


bool
test__log_mvp (void)
{
    const size_t n = 3;
    scalar vt[] = {1, 2, 3};
    scalar mt[] = {1, 2, 3, 2, 3, 1, 3, 2, 1};
    scalar *b1 = VA_SCALAR_EMPTY (n);
    scalar *b2 = VA_SCALAR_EMPTY (n*n);
    scalar *res = VA_SCALAR_EMPTY (n);

    v_rnd (n, vt);
    v_rnd (n*n, mt);
    vi_log (vt, n);
    vi_log (mt, n*n);

    log_mvp (mt, vt, n, b1, b2, res);

    FREE (b1);
    FREE (b2);
    FREE (res);
    return false;
}

bool
test__m_lse_centroid_rows (void)
{
    const size_t n_rows = 4;
    const size_t n_cols = 3;
    const size_t n_elem = n_rows * n_cols;
    scalar vals[] = {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    10, 11, 12};

    scalar *log_vals = VA_SCALAR_EMPTY (n_elem);
    scalar expected[] = { 0, 0, 0 };
    scalar weights[] = {1, 2, 3, 4};
    scalar centroid[] = {0, 0, 0};
    scalar sbuff[] = { 0 , 0, 0};
    bool err = true;

    m_log (vals, n_elem, log_vals);
    for (size_t i = 0; i < 12; i++)
    {
        expected[i%n_cols] += vals[i] * weights[i/n_cols];
        sbuff[i%n_cols] += vals[i];
    }
    for (size_t i = 0; i < n_cols; i++)
    {
        expected[i] /= sbuff[i];
    }
    vi_log (expected, n_cols);
    m_lse_centroid_rows (log_vals, weights, n_rows, n_cols, centroid);

    /*
    puts ("\n");
    for (size_t i = 0; i < n_cols; i++)
        printf ("c[%2zu]: %Lf\t expected[%2zu]: %Lf\n", i,  centroid[i], i, expected[i]);
    puts("\n");
    */
    for (size_t i = 0; i < n_cols; i++)
    {
        err &= ASSERT_EQUAL (centroid[i], expected[i]);
    }
    FREE (log_vals);
    return !err;
}


bool
test__mm_add_s (void)
{
    const size_t n_elem = 10;
    scalar vals[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    scalar out[] = { 0,0,0,0,0,0,0,0,0,0 };
    scalar cval = rnd_int (1, 100);
    bool err = true;

    mm_add_s (vals, vals, 10, cval, out);
    for (size_t i = 0; i < n_elem; i++)
    {
        scalar xxx = vals[i] + vals[i] + cval;
        err &= ASSERT_EQUAL (out[i], xxx);
        /* printf ("\n VAL: %Lf\t OUT: %Lf\t EXPECTED: %LF\t ASSERT: %d", vals[i], out[i], xxx, err); */
    }
    return !err;
}
