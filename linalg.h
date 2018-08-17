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

              
#define fill_vector(vect, value)	for (size_t __idx_i = 0; __idx_i < (vect -> n); __idx_i++)	\
									{															\
										vect -> data[__idx_i] = value;							\
									}


#define fill_matrix(matrix, value)	for (size_t __idx_i = 0; __idx_i < (mat -> n_rows); __idx_i++)      \
									{																	\
										for (size_t __jdx_j = 0; __jdx_j < (mat-> n_cols); __idx_j++)	\
										{																\
											mat->data[__idx_i][__idx_j] = value;						\
										}																\
									}



#define v_d_ADD(vector, value)		for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)        \
									{																	\
										vector->data[__idx_i] += value;									\
									}


#define v_d_SUB(vector, value)		for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)        \
									{																	\
										vector->data[__idx_i] -= value;									\
									}


#define v_d_MUL(vector, value)		for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)        \
									{																	\
										vector->data[__idx_i] *= value;									\
									}


#define v_d_DIV(vector, value)		for (size_t __idx_i = 0; __idx_i < vector -> n; __idx_i++)  		\
									{																	\
										if (value == 0.)												\
										{																\
											fprintf(stderr, "v_d_DIV: zero division encountered");		\
											exit(-1);													\
										}																\
										else															\
										{																\
											vector->data[__idx_i] /= value;								\
										}																\
									}


typedef long scalar scalar;


typedef struct
{
	size_t n;
	scalar *data;
} Vector;


typedef struct
{
	size_t n_rows;
	size_t n_cols;
	scalar **data;
} Matrix;


Vector *NewEmptyVector(size_t n);
Vector *NewVector(size_t n, scalar fill_val);
void free_vector(Vector *vector);
void print_vector(Vector *vector);

Matrix *NewMatrix(size_t n_rows, size_t n_cols, scalar fill_val);
void free_matrix(Matrix *matrix);
void print_matrix(Matrix *matrix);

Vector *v_d_add(Vector *v, scalar s);
Vector *v_d_sub(Vector *v, scalar s);
Vector *v_d_mul(Vector *v, scalar s);
Vector *v_d_div(Vector *v, scalar s);

Vector *v_v_add(Vector *v, Vector *u);
Vector *v_v_sub(Vector *v, Vector *u);
Vector *v_v_mul(Vector *v, Vector *u);
Vector *v_v_div(Vector *v, Vector *u);

Vector *mat_vect_prod(Matrix *A, Vector *b);
Vector *vect_mat_prod(Vector *v, Matrix *A);
Matrix *matmul(Matrix *A, Matrix *B);


#endif    /* LINALG_H */

