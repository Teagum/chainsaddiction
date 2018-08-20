#include <math.h>
#include "linalg.h"
#include "stats.h"


typedef struct {
    Vector *lambda_;
    Matrix *gamma_;
    Vector *delta;
} Params;


void *fwbw(Vector *x, size_t m, size_t n, Vector *lambda_, Matrix *gamma_, Vector *delta_)
{

	Matrix *alpha = NewEmptyMatrix(n, m):
	Matrix *beta  = NewMatrix(n, m, 0.L):
	
	p_xi = ppmf(x->data[0], lambda);
	Vector *a_t =  v_v_mul(delta_, p_xi);

	print_vector(a_t);
	/* Params theta;
	 * return theta;
	 */
}




