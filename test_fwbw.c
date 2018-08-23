#include "hmmcore.h"
#include "linalg.h"


int main()
{	
	const size_t m = 3;
	const size_t n = 8;

	Scalar __x[] = { 15, 24, 33, 42, 51, 12, 13, 14 }; // 51, 12, 13, 14 };
	Vector *v_x = NewVectorFromArray(n, __x);
	Scalar *x = v_dptr(v_x);

	Scalar __lambda[] = { 10, 20, 30 };
	Vector *_lambda = NewVectorFromArray(m, __lambda);

	Scalar __gamma[] = {	.7, .2, .1,
				.1, .7, .2,
				.2, .1, .7 };	
	Matrix *_gamma = NewMatrixFromArray(m, m, __gamma);

	Scalar __delta[] = { .5, .3, .2 };
	Vector *_delta = NewVectorFromArray(m, __delta); 

	printf("IHIHIHIHIHHIH");

	HmmParams *theta = EM(x, n, m, 100, 1.e-6,
				_lambda->data, _gamma->data, _delta->data);
	printf("XXXXXXXXX");
	if (theta == NULL)
		printf("It#s a NuLL-Pointer11111");


	/*
	v_print(theta->lambda_);
	m_print(theta->gamma_);
	v_print(theta->delta_);

*/
	v_free(v_x);
	v_free(_lambda);
	m_free(_gamma);
	v_free(_delta);

	return 0;
}
