#include "dataset.h"


DataSet *
ds_NewEmpty (void)
{
    DataSet *pds = malloc (sizeof pds);
    MA_ASSERT_ALLOC (pds, "Could not allocate dataset.");
    pds->data = NULL;
    pds->size = 0;
    return pds;
}


DataSet *
ds_New (const size_t n_elem)
{
    DataSet *pds = ds_NewEmpty ();
    pds->data = MA_SCALAR_ZEROS (n_elem);
    pds->size = (n_elem);
    return pds;
}


DataSet *
ds_NewFromFile (const char *path)
{
    cnt n_elem = 0;
    FILE *file = Ca_OpenFile (path, "r");
    DataSet *pds = ds_NewEmpty ();

    Ca_CountLines (file, &n_elem);
    if (n_elem == 0)
    {
        fprintf (stderr, "Empty file: %s\n", path);
        ds_FREE (pds);
        return NULL;
    }
    pds->data = MA_SCALAR_EMPTY (n_elem);
    pds->size = Ca_ReadDataFile (file, n_elem, pds->data);

    Ca_CloseFile (file);
    return pds;
}


inline void
ds_set (DataSet *pds, size_t idx, scalar val)
{
    bool err = false;
#ifdef NO_BOUNDS_CHECK
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
#ifdef NO_BOUNDS_CHECK
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
#ifdef _NO_LD_MATH
    const char fmt[] = "[%4zu]%12.5f\n";
#else
    const char fmt[] = "[%4zu]%12.5Lf\n";
#endif
    if (pds->size == 0) {
        puts ("Empty dataset.");
    }
    else
    {
        for (size_t i = 0; i < pds->size; i++)
        {
            scalar val;
            ds_get (pds, i, &val);
            printf (fmt, i, val);
        }
    }
}
