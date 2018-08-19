#include <stdio.h>
#include <stdlib.h>
#include "linalg.h"


int main() { 
	
	Matrix *A = NewMatrix(3, 6, 3);
	Matrix *B = NewMatrix(6, 5, 2);
	Matrix *D = NewMatrix(5, 4, 3);
	
	printf("Matrix multiplication\n");
	Matrix *C = matmul(A, B);
	print_matrix(C);
	
	printf("Matrix vector product\n");
	Vector *x = NewEmptyVector(5);
	fill_vector(x, 1L);
	print_vector(x);	

	Vector *y = mat_vect_prod(B, x);
		
	print_vector(y);
	/*
	printf("Vector matrix product\n");
	y = vect_mat_prod(x, D);
	print_vector(y);

	printf("Vector / scalar operations:\n");
	printf("v_d_add\n");
	fill_vector(x, 5);
	print_vector(v_s_add(x, 5));

	printf("v_d_sub\n");
	print_vector(v_s_sub(x, 5));

	printf("v_d_mul\n");
	print_vector(v_s_mul(x, 5));
	
	printf("v_d_div\n");
	print_vector(v_s_div(x, 5));

	*/	
	free_matrix(A);
	free_matrix(B);
	free_matrix(C);
	free_matrix(D);
	
	free_vector(x);
	return 0;
}

