#include "libma.h"


size_t
Ma_TypeSize (enum ma_types type)
{
    size_t type_size = 0;
    switch (type)
    {
        case MA_SHORT:
            type_size = sizeof (short);
            break;

        case MA_INT:
            type_size = sizeof (int);
            break;

        case MA_FLOAT:
            type_size = sizeof (float);
            break;

        case MA_DOUBLE:
            type_size = sizeof (double);
            break;

        case MA_SCALAR:
            type_size = sizeof (scalar);
            break;
    }
    return type_size;
}


void *
Ma_ArrayMemAlloc (const size_t n_elem, enum ma_types type, bool init)
{
    size_t  size = Ma_TypeSize (type);
    void   *mem  = NULL;

    mem = init ? calloc (n_elem, size) : malloc (n_elem * size);
    MA_ASSERT_ALLOC (mem, "Ma_ArrayMemAlloc: Could not allocate buffer.");
    return mem;
}
