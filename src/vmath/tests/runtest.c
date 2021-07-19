#include "unittest.h"
#include "test_vmath.h"


int main (void)
{
    SETUP;

    RUN_TEST (test__strided_max);
    RUN_TEST (test__v_lse);
    RUN_TEST (test__vs_lse_centroid);
    RUN_TEST (test__v_max);
    RUN_TEST (test__vs_sum);
    RUN_TEST (test__m_max);
    RUN_TEST (test__m_row_max);

    RUN_TEST (test__m_col_max);
    RUN_TEST (test__log_vmp);

    RUN_TEST (test__log_mvp);
    RUN_TEST (test__m_log_centroid_cols);

    EVALUATE;
}
