#include "linalg.h"


Vector *NewEmptyVector(size_t n)
{
	scalar *data = malloc( n * sizeof(scalar) );
	checkVectorAlloc(data);

	Vector *vector = malloc( sizeof(Vector) );
	checkVectorAlloc(vector);

	vector -> n = n;
	vector -> data = data;

	return vector;
}


Vector *NewVector(size_t n, scalar val)
{
	Vector *vector = NewEmptyVector(n);
	fill_vector(vector, val);
	return vector;
}


void free_vector(Vector *vector)
{
	free( vector -> data );
	free( vector );
}


void print_vector(Vector *vector)
{
	for (size_t i = 0; i < vector -> n; i++)
		printf("%f\n", vector -> data[i]);
	printf("\n");
}



Matrix *NewMatrix(size_t n_rows, size_t n_cols, scalar fill_val)
{
	scalar **data = malloc( n_rows * sizeof(scalar *) );
	checkMatrixAlloc(data);

	for (size_t i = 0; i < n_rows; i++)
	{
		data[i] = malloc( n_cols * sizeof(scalar) );
		checkMatrixAlloc(data[i]);
	}
	
	for (size_t i = 0; i < n_rows; i++)
	{
		for (size_t j = 0; j < n_cols; j++)
		{
			data[i][j] = fill_val;
		}
	}

	Matrix *matrix = malloc( sizeof(Matrix) );
	checkMatrixAlloc(matrix);

	matrix -> n_rows = n_rows;
	matrix -> n_cols = n_cols;
	matrix -> data = data;

	return matrix;
}	


void free_matrix(Matrix *matrix)
{
	for (size_t i = 0; i < matrix -> n_rows; i++)
	{
		free( matrix -> data[i] );
	}
	free( matrix -> data );	
	free( matrix );
}


void print_matrix(Matrix *matrix)
{
	for (size_t i = 0; i < matrix -> n_rows; i++)
	{
		for (size_t j = 0; j < matrix -> n_cols; j++)
		{
			printf("%f\t", matrix -> data[i][j]);
		}
		printf("\n");
	}
}

/* 
 * Vector scalar operations 
 */

Vector *v_d_add(Vector *v, scalar s)
{
    Vector *out = NewEmptyVector(v->n);
    
    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] + s;
    }

    return out;
}


Vector *v_d_sub(Vector *v, scalar s)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] - s;
    }

    return out;
}


Vector *v_d_mul(Vector *v, scalar s)
{
    Vector *out = NewEmptyVector(v->n);
    
    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] * s;
    }

    return out;
}


Vector *v_d_div(Vector *v, scalar s)
{
    Vector *out = NewEmptyVector(v->n);
    
    if ( s != 0 )
    {
        for (size_t i = 0; i < (v -> n); i++)
        {
            out -> data[i] = v -> data[i] / s;
        }
    }
    else
    {
        return NULL;
    }

    return out;
}

/*
 * Elementwise vector / vector operations
 */

Vector *v_v_add(Vector *v, Vector *u)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        out->data[i] = v->data[i] + u->data[i];
    }

    return out;
}


Vector *v_v_sub(Vector *v, Vector *u)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        out->data[i] = v->data[i] - u->data[i];
    }

    return out;
}


Vector *v_v_mul(Vector *v, Vector *u)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        out->data[i] = v->data[i] * u->data[i];
    }

    return out;
}


Vector *v_v_div(Vector *v, Vector *u)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        if ( ! (u->data[i] == 0.) )
        {
            out->data[i] = v->data[i] + u->data[i];
        }
        else
        {
            fprintf(stderr, "v_v_div: Argh, Zero division!");
            return NULL;
        }
    }

    return out;
}


Matrix *matmul(Matrix *A, Matrix *B)
{
	Matrix *C = NewMatrix( A -> n_rows, B -> n_cols, 0);

	for (size_t i = 0; i < A -> n_rows; i++)
	{
		for (size_t j = 0; j < B -> n_cols; j++)
		{
			for (size_t k = 0; k < A -> n_cols; k++)
			{
				C -> data[i][j] += ( A -> data[i][k] ) * ( B -> data[k][j] );
			}
		}
	}
	return C;
}

Vector *mat_vect_prod(Matrix *A, Vector *b)
{
	Vector *out = NewEmptyVector(b->n);
	for (size_t i = 0; i < A->n_rows; i++)
	{
		for (size_t j = 0; j < A->n_cols; j++)
		{
			out->data[i] += A->data[i][j] * b->data[j];
		}
	}
	return out;
}

Vector *vect_mat_prod(Vector *v, Matrix *A)
{
	Vector *out = NewEmptyVector(A->n_cols);
	for (size_t i = 0; i < A->n_cols; i++)
	{
		for (size_t j = 0; j < v->n; j++)
		{
			out->data[i] += A->data[j][i] * v->data[j];
		}
	}
	return out;
}
