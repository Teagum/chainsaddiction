#include <stdio.h>
#include <stdlib.h>
#include "linalg.h"


int main() {
	Matrix *A = NewMatrix(3, 6, 3);
	Matrix *B = NewMatrix(6, 5, 2);
	Matrix *D = NewMatrix(5, 4, 3);

	Matrix *C = matmul(A, B);
	print_matrix(C);
	
	Vector *x = NewEmptyVector(5);
	fill_vector(x, 5);
	Vector *y = mat_vect_prod(B, x);
	print_vector(y);

	y = vect_mat_prod(x, D);
	print_vector(y);

	free_matrix(A);
	free_matrix(B);
	free_matrix(C);

	return 0;
}

