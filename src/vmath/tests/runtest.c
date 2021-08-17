#include "unittest.h"
#include "test_vmath.h"
#include "test_rnd.h"


int main (void)
{
    SETUP;

    RUN_TEST (test__rnd_int);
    RUN_TEST (test__v_rnd_int);
    RUN_TEST (test__rnd_scalar);
    RUN_TEST (test__v_rnd_scalar);
    RUN_TEST (test__sample);
    RUN_TEST (test__v_sample);

    RUN_TEST (test__strided_max);
    RUN_TEST (test__v_argmax);
    RUN_TEST (test__v_argmax__max_on_first);
    RUN_TEST (test__v_argmax__max_on_last);
    RUN_TEST (test__v_max);
    RUN_TEST (test__v_lse);
    RUN_TEST (test__vs_lse_centroid);
    RUN_TEST (test__vs_sum);

    RUN_TEST (test__m_max);
    RUN_TEST (test__m_row_argmax);
    RUN_TEST (test__m_row_max);
    RUN_TEST (test__m_col_max);
    RUN_TEST (test__m_log_centroid_cols);

    RUN_TEST (test__log_vmp);
    RUN_TEST (test__log_mvp);

    EVALUATE;
    return 0;
}
