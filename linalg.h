#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <stdio.h>

#define checkVectorAlloc(ptr)	if (ptr == NULL)							\
								{											\
									printf("Vector allocation failed.\n");	\
									return NULL;							\
								}									


#define checkMatrixAlloc(ptr)	if (ptr == NULL)							\
								{											\
									printf("Matrix allocation failed).\n"); \
									return NULL;							\
								}


#define fill_vector(vect, value)	for (size_t i = 0; i < vect -> n; i++)	\
									{										\
										vect -> data[i] = value;			\
									}

#define fill_matrix(matrix, value)	for (size_t i = 0; i < mat -> n_rows; i++)      	\
									{													\
										for (size_t j = 0; j < mat->n_cols; j++)		\
										{												\
											mat->data[i][j] = value;					\
										}												\
									}

#define v_d_sub(vector, value)		for (size_t i = 0; i < vector -> n_rows; i++)		\
									{													\
										vector->data[i] -= value;						\
									}


#define v_d_add(vector, value)		for (size_t i = 0; i < vector -> n_rows; i++)		\
									{													\
										vector->data[i] += value;						\
									}


#define v_d_mul(vector, value)		for (size_t i = 0; i < vector -> n_rows; i++)		\
									{													\
										vector->data[i] *= value;						\
									}

#define v_d_div(vector, value)		for (size_t i = 0; i < vector -> n_rows; i++)		\
									{													\
										if (value == 0.)								\
										{												\
											fprintf(stderr, "Division by zero encountered"); 
											exit\
										}												\
										else											\
										{												\
											vector->data[i] /= value;					\
										}												\
									}


									
typedef struct
{
	size_t n;
	double *data;
} Vector;


typedef struct
{
	size_t n_rows;
	size_t n_cols;
	double **data;
} Matrix;


Vector *NewEmptyVector(size_t n);
Vector *NewVector(size_t n, double fill_val);
void free_vector(Vector *vector);
void print_vector(Vector *vector);


Matrix *NewMatrix(size_t n_rows, size_t n_cols, double fill_val);
void free_matrix(Matrix *matrix);
void print_matrix(Matrix *matrix);

Vector *mat_vect_prod(Matrix *A, Vector *b);
Vector *vect_mat_prod(Vector *v, Matrix *A);
Matrix *matmul(Matrix *A, Matrix *B);

#endif    /* LINALG_H */

