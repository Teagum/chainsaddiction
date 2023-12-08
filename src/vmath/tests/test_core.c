#include "test_core.h"


bool
test__acc_prod (void)
{
    enum setup {
        MAX_VECTOR_SIZE =   1000,
        SR_LB           =  -100,
        SR_UB           =   100
    };

    const size_t n_elem = rnd_size(1, MAX_VECTOR_SIZE);
          scalar res    = 0.0L;
          scalar prod   = 0.0L;

    scalar *vtx = VA_SCALAR_ZEROS (n_elem);
    if (vtx == NULL) { return true; }
    v_rnd_scalar (n_elem, SR_LB, SR_UB, vtx);


    for (size_t i = 0; i < n_elem; i++)
    {
        prod *= vtx[i];
    }
    acc_prod (n_elem, 1, vtx, &res);


    FREE (vtx);
    return ASSERT_EQUAL (res, prod) ? false : true;
}
