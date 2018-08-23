#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#define checkVectorAlloc(ptr)						\
if (ptr == NULL)							\
{									\
	fprintf(stderr, "Vector allocation failed.\n");			\
	return NULL;							\
}									


#define checkMatrixAlloc(ptr)						\
if (ptr == NULL)							\
{									\
	fprintf(stderr, "Matrix allocation failed).\n"); 		\
	return NULL;							\
}

/*
 * New types 
 */

typedef long double Scalar;


typedef struct {
	size_t n;
	Scalar *data;
} Vector;


typedef struct {
	size_t n_rows;
	size_t n_cols;
	Scalar **data;
} Matrix;


/* 
 * Vector inline interface
 */

#define v_FILL(vect, value)						\
for (size_t __idx_i = 0; __idx_i < (vect -> n); __idx_i++)		\
{									\
	vect -> data[__idx_i] = value;					\
}


#define m_FILL(mat, value)						\
for (size_t __idx_i = 0; __idx_i < (mat -> n_rows); __idx_i++)      	\
{									\
	for (size_t __idx_j = 0; __idx_j < (mat-> n_cols); __idx_j++)	\
	{								\
		mat->data[__idx_i][__idx_j] = value;			\
	}								\
}


#define v_s_ADD(vector, value)						\
for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)		\
{									\
	vector->data[__idx_i] += value;					\
}


#define v_s_SUB(vector, value)						\
for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)		\
{									\
	vector->data[__idx_i] -= value;					\
}


#define v_s_MUL(vector, value)						\
for (size_t __idx_i = 0; __idx_i < (vector -> n); __idx_i++)        	\
{									\
	vector->data[__idx_i] *= value;					\
}


#define v_s_DIV(vector, value)						\
for (size_t __idx_i = 0; __idx_i < vector -> n; __idx_i++)  		\
{									\
	if (value == 0.)						\
	{								\
		fprintf(stderr, "v_s_DIV: zero division encountered.");	\
		exit(-1);						\
	}								\
	else								\
	{								\
		vector->data[__idx_i] /= value;				\
	}								\
}


#define v_v_ADD(__v, __u)						\
for (size_t __idx_i = 0; __idx_i < (__v -> n); __idx_i++)		\
{									\
	__v->data[__idx_i] += __u->data[__idx_i];			\
}


#define v_v_SUB(__v, __u)						\
for (size_t __idx_i = 0; __idx_i < (__v -> n); __idx_i++)		\
{									\
	__v->data[__idx_i] -= __u->data[__idx_i];			\
}


#define v_v_MUL(__v, __u)						\
for (size_t __idx_i = 0; __idx_i < (__v -> n); __idx_i++)        	\
{									\
	__v->data[__idx_i] *= __u->data[__idx_i];			\
}


#define v_v_DIV(__v, __u)						\
for (size_t __idx_i = 0; __idx_i < (__v -> n); __idx_i++)  		\
{									\
	if (__u->data[i] == 0L)						\
	{								\
		fprintf(stderr, "v_v_DIV: zero division encountered.");	\
		exit(-1);						\
	}								\
	else								\
	{								\
		__v->data[__idx_i] /= __u->data[__idx_i];		\
	}								\
}




/* 
 * Vector interface
 */

/* Vector initialization */
Vector *NewEmptyVector(size_t n);
Vector *NewVector(size_t n, Scalar fill_val);
Vector *NewVectorFromArray(size_t n, Scalar arr[]);

/* Data access */
Scalar *v_dptr(Vector *v);

/* Elementwise ops */
Vector *v_exp(Vector *v);
Vector *v_m1_exp(Vector *v);
Vector *v_log(Vector *v);
Vector *v_p1_log(Vector *v);

/* Elementwise Vector/Scalar operations */
Vector *v_s_add(Vector *v, Scalar s);
Vector *v_s_sub(Vector *v, Scalar s);
Vector *v_s_mul(Vector *v, Scalar s);
Vector *v_s_div(Vector *v, Scalar s);

/* Elementwise Vector/Vector operations */
Vector *v_v_add(Vector *v, Vector *u);
Vector *v_v_sub(Vector *v, Vector *u);
Vector *v_v_mul(Vector *v, Vector *u);
Vector *v_v_div(Vector *v, Vector *u);

/* Vector reducing ops */
Scalar v_sum(Vector *v);
Scalar v_prod(Vector *v);
Scalar v_mean(Vector *v);

/* Vector deallocation */
void v_free(Vector *vector);

/* Vector output */
void v_print(Vector *vector);

/*
 * Matrix interface
 */

/* Matrix initialization */
Matrix *NewEmptyMatrix(size_t n_rows, size_t n_cols);
Matrix *NewMatrix(size_t n_rows, size_t n_cols, Scalar fill_val);
Matrix *NewMatrixFromArray(size_t m, size_t n, Scalar arr[]);

/* Data access */
Scalar **m_dptr(Matrix *M);
void m_set_row(Matrix *M, size_t i, Vector *v);
void m_set_col(Matrix *M, size_t i, Vector *v);

/* Matrix deallocation */
void m_free(Matrix *matrix);

/* Matrix output */
void m_print(Matrix *matrix);

Vector *m_v_mul(Matrix *A, Vector *b);
Vector *v_m_mul(Vector *v, Matrix *A);
Matrix *m_m_mul(Matrix *A, Matrix *B);


#endif    /* LINALG_H */

