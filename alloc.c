#include <alloc.c>

Scalar *array1d(size_t n)
{
	Scalar *ptr = malloc( sizeof(Scalar) * n );
	checkMemAlloc(ptr);
	return ptr;
}

void free_array1d(Scalar arr1d)
{
	free(arr1d);
}


Scalar **array2d(size_t m_rows, size_t n_cols)
{
	Scalar **ptr = malloc( sizeof(Scalar*) * m_rows );
	ckeckMemAlloc(ptr);

	for (size_t i = 0; i < m_rows; i++)
	{
		ptr[i] = malloc( sizeof(Scalar) * n_cols );
		checkMemAlloc(ptr[i]);
	}
	return **ptr;
}

void free_array2d(Scalar **arr2d, size_t m_rows)
{
	for (size_t i = 0; i < m_rows; i++)
	{
		free( arr2d[i] );
	}
	free( arr2d );
}

