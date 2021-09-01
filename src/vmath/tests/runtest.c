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

    RUN_TEST (test__v_sum);
    RUN_TEST (test__v_sumlog);

    RUN_TEST (test__v_argmax);
    RUN_TEST (test__v_argmax__max_on_first);
    RUN_TEST (test__v_argmax__max_on_last);

    RUN_TEST (test__v_argmin);
    RUN_TEST (test__v_argmin__min_on_first);
    RUN_TEST (test__v_argmin__min_on_last);

    RUN_TEST (test__v_max);
    RUN_TEST (test__v_max__max_on_first);
    RUN_TEST (test__v_max__max_on_last);

    RUN_TEST (test__v_softmax);
    RUN_TEST (test__v_lse);
    RUN_TEST (test__vs_lse_centroid);
    RUN_TEST (test__vs_sum);

    RUN_TEST (test__vm_add);
    RUN_TEST (test__vmi_add);
    RUN_TEST (test__vm_multiply);
    RUN_TEST (test__vm_multiply_log);

    RUN_TEST (test__mv_multiply);
    RUN_TEST (test__mv_multiply_log);

    RUN_TEST (test__m_max);
    RUN_TEST (test__m_row_argmax);
    RUN_TEST (test__m_row_max);
    RUN_TEST (test__m_col_max);
    RUN_TEST (test__m_log_centroid_cols);

    RUN_TEST (test__mm_multiply);

    return 0;
}
