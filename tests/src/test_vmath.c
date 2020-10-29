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
