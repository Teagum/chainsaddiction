#include "matrix.h"


scalar **alloc_matrix(size_t m_rows, size_t n_cols)
{
	scalar **M = malloc( m_rows * sizeof(scalar *) );
	if ( NULL == M ) goto fail;

	for (size_t i = 0; i < m_rows; i++)
	{
		M[i] = malloc( n_cols * sizeof(scalar) );
		if ( NULL == M[i] ) goto fail;
	}
	return M;

fail:
	fprintf(stderr, "Error during matrix allocation.");
	return NULL;
}


void free_matrix(scalar **M, size_t m_rows)
{
	for (size_t i = 0; i < m_rows; i++)
	{
		free( M[i] );
	}

	free( M );
}
