#include "hmmcore.h"
#include "linalg.h"


int main()
{	
	const size_t m = 3;
	const size_t n = 107;

	Scalar __x[] = {  13,14,8,10,16,26,32,27,18,32,36,24,22,23, 
		22,18,25,21,21,14,8,11,14,23,18,17,19,20,22,19,13,26,
		13,14,22,24,21,22,26,21,23,24,27,41,31,27,35,26,28,36,
		39,21,17,22,17,19,15,34,10,15,22,18,15,20,15,22,19,16,
		30,27,29,23,20,16,21,21,25,16,18,15,18,14,10,15,8,15,6,
		11,8,7,18,16,13,12,13,20,15,16,12,18,15,16,13,15,16,11,11};

	Vector *v_x = NewVectorFromArray(n, __x);
	Scalar *x = v_dptr(v_x);

	Scalar __lambda[] = { 10, 20, 30 };
	Vector *_lambda = NewVectorFromArray(m, __lambda);

	Scalar __gamma[] = {	.8, .1, .1,
				.1, .8, .1,
				.1, .1, .8 };	
	Matrix *_gamma = NewMatrixFromArray(m, m, __gamma);

	Scalar __delta[] = { .3333, .3333, .3333 };
	Vector *_delta = NewVectorFromArray(m, __delta); 
	
	HmmParams *theta = EM(x, n, m, 100, 1.e-6,
				_lambda->data, _gamma->data, _delta->data);
	if (theta == NULL)
		printf("It#s a NuLL-Pointer11111");
	

	v_print(theta->lambda_);
	m_print(theta->gamma_);
	v_print(theta->delta_);
	free_HmmParams(theta);

	v_free(v_x);
	v_free(_lambda);
	m_free(_gamma);
	v_free(_delta);

	return 0;
}
