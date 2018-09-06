#include <stdio.h>
#include "matrix.h"

int main(void)
{
	size_t rows = 4;
	size_t cols = 6;

	scalar **p = alloc_matrix(rows, cols);

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			printf("%Lf\t", *(*(p+i)+j) );
		}
		printf("\n");
	}
	free_matrix(p, rows);

	return 0;
}
