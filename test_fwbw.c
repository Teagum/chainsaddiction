#include "linalg.h"
#include "stats.h"


int main()
{
	Vector *x = NewVector(3, 4);
	Vector *lambda = NewVector(3, 3);
	Scalar arr[] = { .5, .3, .2 };
	Vector *delta = NewFromArray(3, arr); 

	Vector *p_xi = ppmf(lambda, x->data[0]);
	Vector *a_t =  v_v_mul(delta, p_xi);
	print_vector(a_t);


	free_vector(x);
	free_vector(lambda);
	free_vector(delta);
	free_vector(p_xi);
	free_vector(a_t);
	return 0;
}
