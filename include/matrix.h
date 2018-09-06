#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>


typedef long double scalar;

scalar	**alloc_matrix	(size_t m_rows, size_t n_cols);
void	free_matrix		(scalar **M, size_t m_rows);

#endif	/* MATRIX_H */
