#include "test_vmath.h"


bool
test__v_sum (void)
{
    enum setup {
        max_vector_size =  1000,
        SR_LB           =  -100,
        SR_UB           =   100
    };

    const size_t n_elem = rnd_size (1, max_vector_size);
          scalar res    = 0.0L;
          scalar sum    = 0.0L;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }
    v_rnd_scalar (n_elem, SR_LB, SR_UB, vtx);


    for (size_t i = 0; i < n_elem; i++)
    {
        sum += vtx[i];
    }
    res = v_sum (n_elem, vtx);

    FREE (vtx);
    return ASSERT_EQUAL (res, sum) ? false : true;
}


bool
test__v_sumlog (void)
{
    enum setup {
        max_vector_size =  1000,
        SR_LB           =     1,
        SR_UB           =   100
    };

    const size_t n_elem = rnd_size (1, max_vector_size);
          scalar res    = 0.0L;
          scalar sum    = 0.0L;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }
    v_rnd_scalar (n_elem, SR_LB, SR_UB, vtx);


    for (size_t i = 0; i < n_elem; i++)
    {
        sum += logl (vtx[i]);
    }
    res = v_sumlog (n_elem, vtx);

    FREE (vtx);
    return ASSERT_EQUAL (res, sum) ? false : true;
}


bool
test__v_argmin (void)
{
    const size_t n_elem  = rnd_size (1, 10);
    const size_t min_idx = rnd_size (0, n_elem-1);
    const scalar min_val = -111.0L;
          size_t res_idx =    0u;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_scalar (n_elem, -10, 10, vtx);
    vtx[min_idx] = min_val;
    res_idx = v_argmin (n_elem, vtx);

    FREE (vtx);
    return (min_idx == res_idx) ? false : true;
}


bool
test__v_argmin__min_on_first (void)
{
    enum setup {
        max_vector_size =  100,
        min_idx         =    0,
        min_val         = -111,
        SR_LB           = -100,
        SR_UB           =  100
    };

    const size_t n_elem  = rnd_size (1, max_vector_size);
          size_t res_idx = 0u;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_scalar (n_elem, SR_LB, SR_UB, vtx);
    vtx[min_idx] = min_val;
    res_idx = v_argmin (n_elem, vtx);

    FREE (vtx);
    return (min_idx == res_idx) ? false : true;
}


bool
test__v_argmin__min_on_last (void)
{
    enum setup {
        max_vector_size =  100,
        min_val         = -111,
        SR_LB           = -100,
        SR_UB           =  100
    };

    const size_t n_elem  = rnd_size (1, max_vector_size);
    const size_t min_idx = n_elem - 1;
          size_t res_idx = 0u;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_scalar (n_elem, SR_LB, SR_UB, vtx);
    vtx[min_idx] = min_val;
    res_idx = v_argmin (n_elem, vtx);

    FREE (vtx);
    return (min_idx == res_idx) ? false : true;
}

bool
test__v_argmax (void)
{
    const size_t n_elem  = rnd_size (1, 10);
    const size_t max_idx = rnd_size (0, n_elem-1);
    const scalar max_val = 999.0L;
          size_t res_idx = 0;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_sample (n_elem, vtx);
    vtx[max_idx] = max_val;
    res_idx = v_argmax (n_elem, vtx);

    FREE (vtx);
    return (max_idx == res_idx) ? false : true;
}


bool test__v_argmax__max_on_first (void)
{
    const size_t n_elem  = rnd_size (1, 10);
    const size_t max_idx = 0;
    const scalar max_val = 999.0L;
          size_t res_idx = 999;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_sample (n_elem, vtx);
    vtx[max_idx] = max_val;
    res_idx = v_argmax (n_elem, vtx);

    FREE (vtx);
    return (max_idx == res_idx) ? false : true;
}


bool test__v_argmax__max_on_last (void)
{
    const size_t n_elem  = rnd_size (1, 10);
    const size_t max_idx = n_elem-1;
    const scalar max_val = 999.0L;
          size_t res_idx = 999;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_sample (n_elem, vtx);
    vtx[max_idx] = max_val;
    res_idx = v_argmax (n_elem, vtx);

    FREE (vtx);
    return (max_idx == res_idx) ? false : true;
}


bool
test__strided_max (void)
{
    bool err = true;
    const size_t n_elem  = rnd_size (20, 100);
    const size_t stride  = rnd_size (1, 20);
    const size_t max_idx = rnd_size (0, n_elem / stride) * stride;
    const scalar max_val = 99.0L;
          scalar max_res =  0.0L;

    scalar *mtx = VA_SCALAR_ZEROS (n_elem);
    if (mtx == NULL) { return err; }

    v_rnd_sample (n_elem, mtx);
    mtx[max_idx] = max_val;

    max_res = strided_max (n_elem, stride, mtx);
    err = !ASSERT_EQUAL (max_res, max_val);

    FREE (mtx);
    return err;
}


bool
test__v_lse (void)
{
    enum setup {
        max_vector_size =  100,
    };

    const size_t n_elem   = rnd_size (1, max_vector_size);
          scalar lsum_vtx = 0;
          scalar lse      = 0;

    scalar *vtx  = VA_SCALAR_EMPTY (n_elem);
    scalar *lvtx = VA_SCALAR_EMPTY (n_elem);
    if (vtx == NULL || lvtx == NULL) return true;

    v_rnd_sample (n_elem, vtx);
    v_log (n_elem, vtx, lvtx);
    for (size_t i = 0; i < n_elem; i++)
    {
        lsum_vtx += vtx[i];
    }
    lsum_vtx = logl (lsum_vtx);
    lse = v_lse (n_elem, lvtx);

    FREE (vtx);
    FREE (lvtx);
    return ASSERT_EQUAL (lsum_vtx, lse) ? false : true;
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

    v_rnd_sample (n_elem, weights);
    v_rnd_sample (n_elem, vals);
    v_log (n_elem, vals, lvals);
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

    v_rnd_sample (n_elem, vals);
    vals[n_elem/2] = max;
    err = !ASSERT_EQUAL (v_max (n_elem, vals), max);

    FREE (vals);
    return err;
}

bool
test__v_max__max_on_first (void)
{
    size_t max_idx = 0u;
    scalar max_val = 2.0L;
    scalar res     = 0.0L;
    size_t n_elem  = rnd_size (1, 100);
    scalar *vtx    = VA_SCALAR_EMPTY (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_sample (n_elem, vtx);
    vtx[max_idx] = max_val;
    res = v_max (n_elem, vtx);

    FREE (vtx);
    return ASSERT_EQUAL (res, max_val) ? false : true;
}


bool
test__v_max__max_on_last (void)
{
    scalar max_val = 2.0L;
    scalar res     = 0.0L;
    size_t n_elem  = rnd_size (1, 100);
    size_t max_idx = n_elem - 1;
    scalar *vtx    = VA_SCALAR_EMPTY (n_elem);
    if (vtx == NULL) { return true; }

    v_rnd_sample (n_elem, vtx);
    vtx[max_idx] = max_val;
    res = v_max (n_elem, vtx);

    FREE (vtx);
    return ASSERT_EQUAL (res, max_val) ? false : true;
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

    v_rnd_sample (n_elem, vals);
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
    enum setup {
        MAX_VAL    =   999,
        N_ROWS_MAX =   100,
        N_COLS_MAX =   100,
        SRANGE_LB  =  -100,
        SRANGE_UB  =   100
    };

    const size_t rows = rnd_size (1, N_ROWS_MAX);
    const size_t cols = rnd_size (1, N_COLS_MAX);
    const size_t midx = rnd_size (0, rows*cols);

    scalar *mtx = VA_SCALAR_EMPTY (rows*cols);
    if (mtx == NULL)
    {
        const char fmt[] = "\n(%s, %d) test__m_max:\nMemory error.";
        fprintf (stderr, fmt, __FILE__, __LINE__);
        VM_RETURN_FAILURE;
    }

    m_rnd_scalar (rows, cols, SRANGE_LB, SRANGE_UB, mtx);
    mtx[midx] = MAX_VAL;

    return !ASSERT_EQUAL (m_max (mtx, rows, cols), MAX_VAL);
}


bool
test__m_row_max (void)
{
    enum setup {
        MIN_ROW_SIZE = 1, MAX_ROW_SIZE = 100,
        MIN_COL_SIZE = 1, MAX_COL_SIZE = 20,
        CHECK_VAL = 10,
        n_repeat_test = 100
    };

    bool is_equal = true;
    for (size_t n = 0; n < n_repeat_test; n++)
    {
        const size_t n_rows = (size_t) rnd_int (MIN_ROW_SIZE, MAX_ROW_SIZE);
        const size_t n_cols = (size_t) rnd_int (MIN_COL_SIZE, MAX_COL_SIZE);
        const size_t n_elem = n_rows * n_cols;

        scalar *mtx = VA_SCALAR_EMPTY (n_elem);
        scalar *res = VA_SCALAR_EMPTY (n_rows);

        m_rnd_sample (n_rows, n_cols, mtx);
        for (size_t row = 0; row < n_rows; row++)
        {
            const size_t max_idx = (size_t) rnd_int (0, n_cols);
            mtx[row*n_cols+max_idx] = (scalar) CHECK_VAL;
        }

        m_row_max (mtx, n_rows, n_cols, res);
        for (size_t i = 0; i < n_rows; i++)
        {
            is_equal &= ASSERT_EQUAL (res[i], (scalar) CHECK_VAL);
        }

        FREE (mtx);
        FREE (res);
        if (!is_equal) return true;
    }
    return false;
}


bool
test__m_col_max (void)
{
    enum test_setup {
        MIN_ROWS = 1, MAX_ROWS = 10,
        MIN_COLS = 1, MAX_COLS = 10,
        MIN_SAMPLE_RANGE = -100,
        MAX_SAMPLE_RANGE =  100
    };

    bool err = true;
    const size_t n_rows = rnd_int (MIN_ROWS, MAX_ROWS);
    const size_t n_cols = rnd_int (MIN_ROWS, MAX_ROWS);
    const size_t n_elem = n_rows * n_cols;

    scalar *mtx = VA_SCALAR_EMPTY (n_elem);
    scalar *max_val = VA_SCALAR_EMPTY (n_cols);
    scalar *max_res = VA_SCALAR_EMPTY (n_cols);
    size_t *max_row_idx = VA_SIZE_EMPTY (n_cols);

    m_rnd_scalar (n_rows, n_cols, MIN_SAMPLE_RANGE, MAX_SAMPLE_RANGE, mtx);
    v_rnd_size (n_cols, 0, n_rows, max_row_idx);
    v_rnd_scalar (n_cols, MAX_SAMPLE_RANGE+1, MAX_SAMPLE_RANGE+100, max_val);

    for (size_t i = 0; i < n_cols; i++)
    {
        size_t idx = max_row_idx[i] * n_cols + i;
        assert (idx < n_elem);
        mtx[idx] = (scalar) max_val[i];
    }

    m_col_max (mtx, n_rows, n_cols, max_res);
    for (size_t i = 0; i < n_cols; i++)
    {
        err &= ASSERT_EQUAL (max_res[i], max_val[i]);
    }

    FREE (mtx);
    FREE (max_val);
    FREE (max_res);
    FREE (max_row_idx);
    return !err;
}


bool
test__m_log_centroid_cols (void)
{
    enum setup {
        MIN_ROWS = 1u, MAX_ROWS = 20u,
        MIN_COLS = 1u, MAX_COLS = 20u,
    };

    bool err = true;
    const scalar SR_LB = 1.0E-10L;
    const scalar SR_UB = 1.0L;
    const size_t n_rows = rnd_int (MIN_ROWS, MAX_ROWS);
    const size_t n_cols = rnd_int (MIN_COLS, MAX_COLS);
    const size_t n_elem = n_rows * n_cols;

    scalar *arr             = VA_SCALAR_EMPTY (n_elem);
    scalar *larr            = VA_SCALAR_EMPTY (n_elem);
    scalar *weight_per_row  = VA_SCALAR_EMPTY (n_rows);
    scalar *average_per_col = VA_SCALAR_ZEROS (n_cols);
    scalar *sum_per_col     = VA_SCALAR_ZEROS (n_cols);
    scalar *expected_result = VA_SCALAR_ZEROS (n_cols);

    if (arr == NULL || larr == NULL || weight_per_row == NULL ||
        average_per_col == NULL || sum_per_col == NULL ||
        expected_result == NULL)
    {
        fputs ("Allocation error in ``test__m_log_centroid_cols''.\n", stderr);
        err = true;
        goto exit;
    }

    v_rnd_scalar (n_rows, 0, 100, weight_per_row);
    m_rnd_scalar (n_rows, n_cols, SR_LB, SR_UB, arr);
    m_log (n_rows, n_cols, arr, larr);

    for (size_t i = 0; i < n_elem; i++)
    {
        size_t col_idx = i % n_cols;
        size_t row_idx = i / n_cols;

        expected_result[col_idx] += arr[i] * weight_per_row[row_idx];
        sum_per_col[col_idx] += arr[i];
    }

    for (size_t i = 0; i < n_cols; i++)
    {
        expected_result[i] /= sum_per_col[i];
    }

    vi_log (n_cols, expected_result);
    m_log_centroid_cols (larr, weight_per_row, n_rows, n_cols, average_per_col);

    for (size_t i = 0; i < n_cols; i++)
    {
        if (ASSERT_EQUAL (average_per_col[i], expected_result[i]))
        {
            err = false;
        }
        else
        {
            err = true;
            break;
        }
    }

exit:
    free (arr);
    free (larr);
    free (weight_per_row);
    free (average_per_col);
    free (sum_per_col);
    free (expected_result);
    return err;
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


/*
bool
test__v_argmax (void)
{
    enum {
        size_bound = 20,
    };

    size_t res = 0;
    const size_t n_elem  = rnd_size (1, size_bound);
    const size_t arg_max = rnd_size (0, n_elem-1);
    const scalar max_val =  999.0L;
    const scalar SR_LB   = -100.0L;
    const scalar SR_UB   =  100.0L;

    scalar *vect = VA_SCALAR_ZEROS(n_elem);
    v_rnd_scalar (n_elem, SR_LB, SR_UB, vect);
    vect[arg_max] = max_val;

    res = v_argmax (n_elem, vect);

    print_vector (n_elem, vect);
    printf ("Arg: %zu Expected %zu\n", res, arg_max);
    return res == arg_max ? false : true;
}
*/

bool
test__m_row_argmax (void)
{
    bool err = true;
    const size_t rows = rnd_size (1, 10);
    const size_t cols = rnd_size (1, 10);
    const scalar max_val =  999.0L;
    const scalar SR_LB   = -100.0L;
    const scalar SR_UB   =  100.0L;

    size_t *arg_max = VA_SIZE_ZEROS (rows);
    size_t *res = VA_SIZE_ZEROS (rows);
    scalar *mtx = VA_SCALAR_ZEROS (rows*cols);

    v_rnd_scalar (rows*cols, SR_LB, SR_UB, mtx);
    v_rnd_size (rows, 0, cols, arg_max);

    for (size_t i = 0; i < rows; i++)
    {
        mtx[i*cols+arg_max[i]] = max_val;
    }

    m_row_argmax (rows, cols, mtx, res);


    for (size_t i = 0; i < rows; i++)
    {
        err = (res[i] != arg_max[i]) ? true : false;
        if (err) break;
    }

    FREE (arg_max);
    FREE (res);
    FREE (mtx);
    return err;
}


bool
test__v_softmax (void)
{
    enum setup {
        VECTOR_MAX_SIZE = 1000,
        SRANGE_LB = -10,
        SRANGE_UB =  10
    };
    scalar n_elem = rnd_size (1, VECTOR_MAX_SIZE);
    scalar total  = 0.0L;

    scalar *vtx = VA_SCALAR_EMPTY (n_elem);
    scalar *res = VA_SCALAR_EMPTY (n_elem);
    if (vtx == NULL || res == NULL) VM_RETURN_FAILURE;

    v_rnd_scalar (n_elem, SRANGE_LB, SRANGE_UB, vtx);
    v_softmax (n_elem, vtx, res);
    total = v_sum (n_elem, res);

    FREE (vtx);
    FREE (res);
    return ASSERT_EQUAL (1.0, total) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__vi_softmax (void)
{
    enum setup {
        VECTOR_MAX_SIZE = 5,
        SRANGE_LB = -10,
        SRANGE_UB =  10
    };
    scalar n_elem = rnd_size (1, VECTOR_MAX_SIZE);
    scalar total  = 0.0L;

    scalar *vtx = VA_SCALAR_EMPTY (n_elem);
    if (vtx == NULL) VM_RETURN_FAILURE;

    v_rnd_scalar (n_elem, SRANGE_LB, SRANGE_UB, vtx);
    vi_softmax (n_elem, vtx);
    total = v_sum (n_elem, vtx);

    FREE (vtx);
    return ASSERT_EQUAL (1.0, total) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__vm_add (void)
{
    enum setup {
        N_ROWS_MIN =     1,
        N_ROWS_MAX =   100,
        N_COLS_MIN =     1,
        N_COLS_MAX =   100,
        SRANGE_LB  = -1000,
        SRANGE_UB  =  1000
    };

    bool   err  = false;
    size_t rows = rnd_size (N_ROWS_MIN, N_ROWS_MAX);
    size_t cols = rnd_size (N_COLS_MIN, N_COLS_MAX);

    scalar *vtx = VA_SCALAR_EMPTY (cols);
    scalar *mtx = VA_SCALAR_EMPTY (rows*cols);
    scalar *res = VA_SCALAR_EMPTY (rows*cols);
    scalar *xpc = VA_SCALAR_EMPTY (rows*cols);
    if (vtx == NULL || mtx == NULL)
    {
        const char fmt[] = "(%s, %d)\ntest__vm_add: could not allocate.\n";
        fprintf (stderr, fmt, __FILE__, __LINE__);
        VM_RETURN_FAILURE;
    }

    v_rnd_scalar (cols, SRANGE_LB, SRANGE_UB, vtx);
    m_rnd_scalar (rows, cols, SRANGE_LB, SRANGE_UB, mtx);
    vm_add (rows, cols, vtx, mtx, res);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            size_t idx = i * cols + j;
            xpc[idx] = vtx[j] + mtx[idx];
        }
    }

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            size_t idx = i * cols + j;
            if (!ASSERT_EQUAL (res[idx], xpc[idx]))
            {
                err = true;
                goto exit_test;
            }
        }
    }

exit_test:
    FREE (vtx);
    FREE (mtx);
    FREE (res);
    FREE (xpc);
    return err ? VM_FAILURE : VM_SUCCESS;
}


bool
test__vmi_add (void)
{
    enum setup {
        N_ROWS_MIN =     1,
        N_ROWS_MAX =   100,
        N_COLS_MIN =     1,
        N_COLS_MAX =   100,
        SRANGE_LB  = -1000,
        SRANGE_UB  =  1000
    };

    bool   err  = false;
    size_t rows = rnd_size (N_ROWS_MIN, N_ROWS_MAX);
    size_t cols = rnd_size (N_COLS_MIN, N_COLS_MAX);

    scalar *vtx = VA_SCALAR_EMPTY (cols);
    scalar *mtx = VA_SCALAR_EMPTY (rows*cols);
    scalar *xpc = VA_SCALAR_EMPTY (rows*cols);
    if (vtx == NULL || mtx == NULL)
    {
        const char fmt[] = "(%s, %d)\ntest__vm_add: could not allocate.\n";
        fprintf (stderr, fmt, __FILE__, __LINE__);
        VM_RETURN_FAILURE;
    }

    v_rnd_scalar (cols, SRANGE_LB, SRANGE_UB, vtx);
    m_rnd_scalar (rows, cols, SRANGE_LB, SRANGE_UB, mtx);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            size_t idx = i * cols + j;
            xpc[idx] = vtx[j] + mtx[idx];
        }
    }

    vmi_add (rows, cols, vtx, mtx);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            size_t idx = i * cols + j;
            if (!ASSERT_EQUAL (mtx[idx], xpc[idx]))
            {
                err = true;
                goto exit_test;
            }
        }
    }

exit_test:
    FREE (vtx);
    FREE (mtx);
    FREE (xpc);
    return err ? VM_FAILURE : VM_SUCCESS;
}


bool
test__vm_multiply (void)
{
    enum setup {
        max_rows = 1000,
        max_cols = 1000,
    };

    scalar total = 0.0L;
    size_t rows = rnd_size (1, max_rows);
    size_t cols = rnd_size (1, max_cols);

    scalar *vtx = VA_SCALAR_EMPTY (rows);
    scalar *mtx = VA_SCALAR_EMPTY (rows*cols);
    scalar *res = VA_SCALAR_EMPTY (cols);
    if (vtx == NULL || mtx == NULL || res == NULL) VM_RETURN_FAILURE;

    v_rnd_scalar (rows, 0, 1, vtx);
    v_rnd_scalar (rows*cols, 0, 1, mtx);

    vi_softmax (rows, vtx);
    mi_row_apply (rows, cols, vi_softmax, mtx);

    vm_multiply (rows, cols, vtx, mtx, res);
    total = v_sum (cols, res);

    FREE (vtx);
    FREE (mtx);
    FREE (res);
    return ASSERT_EQUAL (total, 1.0L) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__vm_multiply_log (void)
{
    const size_t rows   = rnd_size (1, 10);
    const size_t cols   = rnd_size (1, 20);
    const size_t n_elem = rows * cols;
          scalar total  = 0.0L;

    scalar *vtx = VA_SCALAR_ZEROS (rows);
    scalar *mtx = VA_SCALAR_ZEROS (n_elem);
    scalar *acc = VA_SCALAR_ZEROS (rows);
    scalar *res = VA_SCALAR_ZEROS (cols);
    if (vtx == NULL || mtx == NULL || acc == NULL || res == NULL)
        VM_RETURN_FAILURE;

    v_rnd_scalar (rows, 1, 10, vtx);
    v_rnd_scalar (n_elem, 1, 10, mtx);

    vi_softmax (rows, vtx);
    mi_row_apply (rows, cols, vi_softmax, mtx);

    vi_log (rows, vtx);
    vi_log (n_elem, mtx);

    vm_multiply_log (rows, cols , vtx, mtx, acc, res);
    vi_exp (cols, res);
    total = v_sum (cols, res);

    FREE (vtx);
    FREE (mtx);
    FREE (acc);
    FREE (res);
    return ASSERT_EQUAL (1.0L, total) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__mv_multiply (void)
{
    enum setup {
        N_ROWS_MAX = 1000,
        N_COLS_MAX = 1000,
        SRANGE_LB  = -100,
        SRANGE_UB  =  100,
    };

    const size_t rows   = rnd_size (1, N_ROWS_MAX);
    const size_t cols   = rnd_size (1, N_COLS_MAX);
    const size_t n_elem = rows * cols;
          scalar total_res  = 0.0L;
          scalar total_mtx  = 0.0L;

    scalar *vtx = VA_SCALAR_ZEROS (cols);
    scalar *mtx = VA_SCALAR_ZEROS (n_elem);
    scalar *res = VA_SCALAR_ZEROS (rows);
    if (vtx == NULL || mtx == NULL || res == NULL)
        VM_RETURN_FAILURE;

    for (size_t i = 0; i < cols; i++)
    {
        vtx[i] = 1.0L;
    }
    v_rnd_scalar (n_elem, SRANGE_LB, SRANGE_UB, mtx);
    mv_multiply (rows, cols, mtx, vtx, res);

    total_res = v_sum (rows, res);
    total_mtx = v_sum (n_elem, mtx);

    FREE (vtx);
    FREE (mtx);
    FREE (res);

    return ASSERT_EQUAL (total_res, total_mtx) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__mv_multiply_log (void)
{
    enum setup {
        N_ROWS_MAX =  1000,
        N_COLS_MAX =  1000,
        SRANGE_LB  = -1000,
        SRANGE_UB  =  1000,
    };

    const size_t rows   = rnd_size (1, N_COLS_MAX);
    const size_t cols   = rnd_size (1, N_ROWS_MAX);
    const size_t n_elem = rows * cols;
          scalar t_xpc  = 0.0L;
          scalar t_res  = 0.0L;

    scalar *vtx = VA_SCALAR_EMPTY (cols);
    scalar *mtx = VA_SCALAR_EMPTY (n_elem);
    scalar *acc = VA_SCALAR_EMPTY (cols);
    scalar *res = VA_SCALAR_EMPTY (rows);
    scalar *xpc = VA_SCALAR_EMPTY (rows);
    if (vtx == NULL || mtx == NULL || acc == NULL || res == NULL ||
        xpc == NULL)
    {
        VM_RETURN_FAILURE;
    }

    v_rnd_scalar (cols, SRANGE_LB, SRANGE_UB, vtx);
    v_rnd_scalar (n_elem, SRANGE_LB, SRANGE_UB, mtx);
    vi_softmax (cols, vtx);
    mi_row_apply (rows, cols, vi_softmax, mtx);

    mv_multiply (rows, cols, mtx, vtx, xpc);
    vi_log (rows, xpc);
    t_xpc = v_sum (rows, xpc);

    vi_log (cols, vtx);
    vi_log (n_elem, mtx);

    mv_multiply_log (rows, cols , mtx, vtx, acc, res);
    t_res = v_sum (rows, res);

    FREE (vtx);
    FREE (mtx);
    FREE (acc);
    FREE (res);
    FREE (xpc);
    return ASSERT_EQUAL (t_xpc, t_res) ? VM_SUCCESS : VM_FAILURE;
}


bool
test__mm_multiply (void)
{
    enum setup {
        rc_max = 100,
    };

    scalar total = 0.0L;
    const size_t rc = rnd_size (2, rc_max);
    const size_t n_elem = rc * rc;

    scalar *A = VA_SCALAR_ZEROS (n_elem);
    scalar *B = VA_SCALAR_ZEROS (n_elem);
    scalar *C = VA_SCALAR_ZEROS (n_elem);
    if (A == NULL || B == NULL || C == NULL) VM_RETURN_FAILURE;

    v_rnd_scalar (n_elem, 0, 1, A);
    v_rnd_scalar (n_elem, 0, 1, B);

    mi_row_apply (rc, rc, vi_softmax, A);
    mi_row_apply (rc, rc, vi_softmax, B);

    mm_multiply (rc, rc, rc, A, B, C);
    total = v_sum (n_elem, C);

    FREE (A);
    FREE (B);
    FREE (C);
    return ASSERT_EQUAL (total, rc) ? VM_SUCCESS : VM_FAILURE;
}
