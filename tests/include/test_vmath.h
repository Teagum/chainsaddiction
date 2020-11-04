#ifndef TEST_VMATH_H
#define TEST_VMATH_H

#include "unittest.h"
#include "scalar.h"
#include "restrict.h"
#include "rnd.h"
#include "vmath.h"

bool test_strided_max (void);

bool test_v_lse (void);
bool test_v_lse_centroid (void);
bool test_v_max (void);

bool test_vs_sum (void);

bool test_m_max (void);
bool test_m_row_max (void);
bool test_m_col_max (void);

bool test_log_vmp (void);
bool test_log_mvp (void);
/*bool test_v_log_centroids (void);*/


#endif  /* TEST_VMATH_H */
