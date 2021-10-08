#include "dataset.h"


DataSet *
ds_NewEmpty (void)
{
    DataSet *this = malloc (sizeof this);
    if (this == NULL)
    {
        Ca_ErrMsg ("Could not allocate data set.");
    }
    else
    {
        this->data = NULL;
        this->size = 0;
        this->err  = false;
    }
    return this;
}


DataSet *
ds_New (const size_t n_elem)
{
    DataSet *this = ds_NewEmpty ();
    this->data = MA_SCALAR_ZEROS (n_elem);
    this->size = (n_elem);
    return this;
}


DataSet *
ds_NewFromFile (const char *path)
{
    size_t n_elem = 0;
    FILE   *file  = Ca_OpenFile (path, "r");
    DataSet *this  = ds_NewEmpty ();

    Ca_CountLines (file, &n_elem);
    if (n_elem == 0)
    {
        fprintf (stderr, "Empty file: %s\n", path);
        ds_FREE (this);
        Ca_CloseFile (file);
        return NULL;
    }
    this->data = MA_SCALAR_EMPTY (n_elem);
    this->size = Ca_ReadDataFile (file, n_elem, this->data);

    Ca_CloseFile (file);
    return this;
}


inline void
ds_set (DataSet *this, size_t idx, scalar val)
{
    bool err = false;
#ifdef NO_BOUNDS_CHECK
    this->data[idx] = val;
#else
    if (idx >= this->size) {
        err = true;
    } else {
        this->data[idx] = val;
        err = false;
    }
#endif
    this->err = err;
}


inline void
ds_get (DataSet *this, size_t idx, scalar *out)
{
    bool err = false;
#ifdef NO_BOUNDS_CHECK
    *out = this->data[idx];
#else
    if (idx >= this->size) {
        err = true;
    } else {
        *out = this->data[idx];
        err = false;
    }
#endif
    this->err = err;
}


extern void
ds_print (DataSet *this)
{
#ifdef _NO_LD_MATH
    const char fmt[] = "[%4zu]%12.5f\n";
#else
    const char fmt[] = "[%4zu]%12.5Lf\n";
#endif
    if (this->size == 0) {
        puts ("Empty dataset.");
    }
    else
    {
        for (size_t i = 0; i < this->size; i++)
        {
            scalar val;
            ds_get (this, i, &val);
            printf (fmt, i, val);
        }
    }
}
