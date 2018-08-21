#include "fwbw.h"
#include "linalg.h"


int main()
{	
	const size_t m = 3;
	const size_t n = 5;

	Scalar __x[] = { 15, 24, 33, 42, 51 };
	Vector *x = NewVectorFromArray(5, __x);

	Scalar __lambda[] = { 10, 20, 30 };
	Vector *_lambda = NewVectorFromArray(m, __lambda);

	Scalar __gamma[] = { .7, .2, .1,
       		             .1, .7, .2,
		 	     .2, .1, .7 };	
	Matrix *_gamma = NewMatrixFromArray(m, m, __gamma);

	Scalar __delta[] = { .5, .3, .2 };
	Vector *_delta = NewVectorFromArray(m, __delta); 



	fwbw(x, m, n, _lambda, _gamma, _delta);


	free_vector(x);
	free_vector(_lambda);
	free_matrix(_gamma);
	free_vector(_delta);
	return 0;
}
