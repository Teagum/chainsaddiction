#include "linalg.h"

/*
 * Vector initialization 
 */

Vector *NewEmptyVector(size_t n)
{
	Scalar *data = malloc( n * sizeof(Scalar) );
	checkVectorAlloc(data);

	Vector *vector = malloc( sizeof(Vector) );
	checkVectorAlloc(vector);

	vector -> n = n;
	vector -> data = data;

	return vector;
}


Vector *NewVector(size_t n, Scalar val)
{
	Vector *vector = NewEmptyVector(n);
	v_FILL(vector, val);
	return vector;
}


Vector *NewVectorFromArray(size_t n, Scalar arr[])
{
	Vector *out = NewEmptyVector(n);
	for (size_t i = 0; i < n; i++)
	{
		out->data[i] = arr[i];
	}
	return out;
}


/*
 * Data access
 */


Scalar *v_dptr(Vector *v)
{
	return v->data;
}


/*
 * Elementwise ops
 */

Vector *v_exp(Vector *v)
{
	Vector *out = NewEmptyVector(v->n);
	for (size_t i = 0; i < (v->n); i++)
	{
		out->data[i] = expl(v->data[i]);
	}
	return out;
}


Vector *v_m1_exp(Vector *v)
{
	Vector *out = NewEmptyVector(v->n);
	for (size_t i = 0; i < (v->n); i++)
	{
		out->data[i] = expm1l(v->data[i]);
	}
	return out;
}


Vector *v_log(Vector *v)
{
	Vector *out = NewEmptyVector(v->n);
	for (size_t i = 0; i < (v->n); i++)
	{
		out->data[i] = logl(v->data[i]);
	}
	return out;
}


Vector *v_p1_log(Vector *v)
{
	Vector *out = NewEmptyVector(v->n);
	for (size_t i = 0; i < (v->n); i++)
	{
		out->data[i] = log1pl(v->data[i]);
	}
	return out;
}


/* 
 * Elementwise Vector / Scalar ops 
 */

Vector *v_s_add(Vector *v, Scalar s)
{
    Vector *out = NewEmptyVector(v->n);
    
    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] + s;
    }

    return out;
}


Vector *v_s_sub(Vector *v, Scalar s)
{
    Vector *out = NewEmptyVector(v->n);

    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] - s;
    }

    return out;
}


Vector *v_s_mul(Vector *v, Scalar s)
{
    Vector *out = NewEmptyVector(v->n);
    
    for (size_t i = 0; i < (v -> n); i++)
    {
        out -> data[i] = v -> data[i] * s;
    }

    return out;
}


Vector *v_s_div(Vector *v, Scalar s)
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
 * Elementwise Vector / Vector ops
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


/*
 * Vector reducing ops
 */

Scalar v_sum(Vector *v)
{
	Scalar vsum = 0;
	for (size_t i = 0; i < (v->n); i++)
	{
		vsum += v->data[i];
	}
	return vsum;
}


Scalar v_prod(Vector *v)
{
	Scalar vprod = 1L;
	for (size_t i = 0; i < (v -> n); i++)
	{
		vprod *= v->data[i];
	}
	return vprod;
}


Scalar v_mean(Vector *v)
{
	return v_sum(v) / (v -> n);
}


Scalar v_max(Vector *v)
{
	Scalar max = v->data[0];
	for (size_t i = 1; i < (v->n); i++)
	{
		max = ( (v->data[i]) > max )  ?  v->data[i]  :  max;
	}
	return max;
}


/*
 * Vector deallocation
 */

void v_free(Vector *vector)
{
	free( vector -> data );
	free( vector );
}


/*
 * Vector output
 */

void v_print(Vector *vector)
{
	fprintf(stdout, "[");
	for (size_t i = 0; i < vector -> n; i++)
		fprintf(stdout, "\t%Lf", vector -> data[i]);
	fprintf(stdout, "\t]\n");
}


/* 
 * Matrix interface
 */

/*
 * Matrix initialization
 */

Matrix *NewEmptyMatrix(size_t n_rows, size_t n_cols)
{
	Scalar **data = malloc( n_rows * sizeof(Scalar *) );
	checkMatrixAlloc(data);

	for (size_t i = 0; i < n_rows; i++)
	{
		data[i] = malloc( n_cols * sizeof(Scalar) );
		checkMatrixAlloc(data[i]);
	}

	Matrix *M = malloc( sizeof(Matrix) );
	checkMatrixAlloc(M);

	M->n_rows = n_rows;
	M->n_cols = n_cols;
	M->data = data;

	return M;
}	


Matrix *NewMatrix(size_t n_rows, size_t n_cols, Scalar fill_val)
{
	Matrix *M = NewEmptyMatrix(n_rows, n_cols);
	m_FILL(M, fill_val);	
	return M;
}



Matrix *NewMatrixFromArray(size_t m, size_t n, Scalar arr[])
{	
	Matrix *out = NewEmptyMatrix(m, n);
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			out->data[i][j] = arr[i*m+j];
		}
	}
	return out;
}


/* 
 * Data access
 */


Scalar **m_dptr(Matrix *M)
{
	return M->data;
}


void m_set_row(Matrix *M, size_t i, Vector *v)
{
	for (size_t j = 0; j < (v->n); j++)
	{
		M->data[i][j] = v->data[j];
	}
}


void m_set_col(Matrix *M, size_t j, Vector *v)
{
	for (size_t i = 0; i < (v->n); i++)
	{
		M->data[i][j] = v->data[i];
	}
}


/*
 * Matrix multiplication
 */

Matrix *m_m_mul(Matrix *A, Matrix *B)
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

Vector *m_v_mul(Matrix *A, Vector *b)
{
	Vector *out = NewEmptyVector(A->n_rows);
	for (size_t i = 0; i < A->n_rows; i++)
	{
		for (size_t j = 0; j < A->n_cols; j++)
		{
			out->data[i] += (A->data[i][j]) * (b->data[j]);
		}
	}
	return out;
}

Vector *v_m_mul(Vector *v, Matrix *A)
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


/*
 * Matrix deallocation
 */

void m_free(Matrix *matrix)
{
	for (size_t i = 0; i < matrix -> n_rows; i++)
	{
		free( matrix -> data[i] );
	}
	free( matrix -> data );	
	free( matrix );
}


/* 
 * Matrix output 
 */

void m_print(Matrix *matrix)
{
	fprintf(stdout, "[\n");
	for (size_t i = 0; i < matrix -> n_rows; i++)
	{
		fprintf(stdout, "\t[");
		for (size_t j = 0; j < matrix -> n_cols; j++)
		{
			fprintf(stdout, "\t%Lf", matrix -> data[i][j]);
		}
		fprintf(stdout, "\t]\n");
	}
	fprintf(stdout, "]\n");
}
