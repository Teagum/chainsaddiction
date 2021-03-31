#include "dataset.h"


DataSet *
Ca_NewDataSet (void)
{
    DataSet *pds = malloc (sizeof pds);
    MA_ASSERT_ALLOC (pds, "Could not allocate dataset.");
    pds->data = MA_SCALAR_ZEROS (DATASET_INIT_SIZE);
    pds->size = DATASET_INIT_SIZE;
    return pds;
}


inline void
ds_set (DataSet *pds, size_t idx, scalar val)
{
    bool err = false;
#ifdef no_bounds_check
    pds->data[idx] = val;
#else
    if (idx >= pds->size) {
#ifndef no_diagnostics
        fprintf (stderr, OUT_OF_BOUNDS_ERR_MSG, idx, pds->size);
#endif
        err = true;
    } else {
        pds->data[idx] = val;
        err = false;
    }
#endif
    pds->err = err;
}


inline void
ds_get (DataSet *pds, size_t idx, scalar *out)
{
    bool err = false;
#ifdef no_bounds_check
    *out = pds->data[idx];
#else
    if (idx >= pds->size) {
#ifndef no_diagnostics
        fprintf (stderr, OUT_OF_BOUNDS_ERR_MSG, idx, pds->size);
#endif
        err = true;
    } else {
        *out = pds->data[idx];
        err = false;
    }
#endif
    pds->err = err;
}
