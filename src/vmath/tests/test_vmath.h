#ifndef test_vmath_h
#define test_vmath_h

#include <assert.h>
#include <stdbool.h>
#include "config.h"
#include "assertations.h"
#include "rnd.h"
#include "vmath.h"


bool test__strided_max (void);

bool test__v_sum (void);
bool test__v_sumlog (void);

bool test__v_argmax (void);
bool test__v_argmax__max_on_first (void);
bool test__v_argmax__max_on_last (void);

bool test__v_argmin (void);
bool test__v_argmin__min_on_first (void);
bool test__v_argmin__min_on_last (void);

bool test__v_softmax (void);

bool test__v_max (void);
bool test__v_max__max_on_first (void);
bool test__v_max__max_on_last (void);
bool test__v_lse (void);
bool test__vs_lse_centroid (void);
bool test__vs_sum (void);
bool test__m_max (void);
bool test__m_row_max (void);
bool test__m_col_max (void);
bool test__log_mvp (void);
bool test__m_log_centroid_cols (void);
bool test__mm_add_s (void);
bool test__m_row_argmax (void);

bool test__vi_softmax (void);

bool test__vm_add (void);
bool test__vmi_add (void);
bool test__vm_multiply (void);
bool test__vm_multiply_log (void);

bool test__mv_multiply (void);
bool test__mv_multiply_log (void);

bool test__mm_multiply (void);
#endif  /* test_vmath_h */
