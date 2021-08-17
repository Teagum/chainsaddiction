#ifndef test_vmath_h
#define test_vmath_h

#include "config.h"
#include "unittest.h"
#include "rnd.h"
#include "vmath.h"


bool test__strided_max (void);
bool test__v_argmax (void);
bool test__v_argmax__max_on_first (void);
bool test__v_argmax__max_on_last (void);
bool test__v_max (void);
bool test__v_max__max_on_first (void);
bool test__v_max__max_on_last (void);
bool test__v_lse (void);
bool test__vs_lse_centroid (void);
bool test__vs_sum (void);
bool test__m_max (void);
bool test__m_row_max (void);
bool test__m_col_max (void);
bool test__log_vmp (void);
bool test__log_mvp (void);
bool test__m_log_centroid_cols (void);
bool test__mm_add_s (void);
bool test__m_row_argmax (void);


#endif  /* test_vmath_h */
