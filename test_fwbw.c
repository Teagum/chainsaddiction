#include "fwbw.h"
#include "linalg.h"


int main()
{	
	const size_t m = 3;
	const size_t n =7;

	Scalar __x[] = { 15, 24, 33, 42, 12, 13, 14 }; // 51, 12, 13, 14 };
	Vector *x = NewVectorFromArray(n, __x);

	Scalar __lambda[] = { 10, 20, 30 };
	Vector *_lambda = NewVectorFromArray(m, __lambda);

	Scalar __gamma[] = { .7, .2, .1,
       		             .1, .7, .2,
				 	     .2, .1, .7 };	
	Matrix *_gamma = NewMatrixFromArray(m, m, __gamma);

	Scalar __delta[] = { .5, .3, .2 };
	Vector *_delta = NewVectorFromArray(m, __delta); 



	AB_probs *ab = fwbw(x->data, n, m, _lambda->data, _gamma->data, _delta->data);

	m_print(ab->alpha);
	m_print(ab->beta);

	v_free(x);
	v_free(_lambda);
	m_free(_gamma);
	v_free(_delta);

	m_free(ab->alpha);
	m_free(ab->beta);
	
	return 0;
}
