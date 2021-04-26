#include "dataset.h"


DataSet *
Ca_NewDataSet (void)
{
    DataSet *pds = malloc (sizeof pds);
    MA_ASSERT_ALLOC (pds, "Could not allocate dataset.");
    pds->data = NULL;
    pds->size = 0;
    return pds;
}


DataSet *
Ca_DataSetFromFile (const char *path)
{
    cnt n_elem = 0;
    cnt r_elem = 0;
    FILE *file = Ca_OpenFile (path, "r");
    DataSet *pds = Ca_NewDataSet ();

    Ca_CountLines (file, &n_elem);
    pds->data = MA_SCALAR_EMPTY (n_elem);
    pds->size = Ca_ReadDataFile (file, n_elem, pds->data);

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
        err = true;
    } else {
        *out = pds->data[idx];
        err = false;
    }
#endif
    pds->err = err;
}


extern void
ds_print (DataSet *pds)
{
    if (pds->size == 0) {
        puts ("Empty dataset.");
    }
    else
    {
        for (size_t i = 0; i < pds->size; i++)
        {
            scalar val;
            ds_get (pds, i, &val);
            printf ("[%4zu]%12.5Lf\n", i, val);
        }
    }
}
