#include <stdlib.h>

Scalar *array1d(size_t n);
Scalar *array2d(size_t m_rows, size_t n_cols);

void free_array1d(Scalar arr1d, size_t n);
void free_array2d(Scalar arr2d, size_t n_rows);

#define checkMemAlloc(ptr)				\
	if ( ptr == NULL )				\
	{						\
		fprintf(stderr, "Alloc Error.");	\
		return NULL;				\
	}
