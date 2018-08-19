#include <math.h>
#include "linalg.h"
#include "stats.h"


typedef struct {
    Vector *lambda_;
    Matrix *gamma_;
    Vector *delta;
} Params;


Params *fwbw(Vector *x, size_t m, size_t n, Vector *lambda_, Matrix *gamma_, Vector *delta_)
{
    
    Matrix *alpha = NewEmptyMatrix(n, m):
    Matrix *beta  = NewMatrix(n, m, 0.L):
	
	/* poisson_pmf should take an vector of 
	 * parameters 
	 */
	
	p_xi = poisson_pmf
    Vector *a_t =  v_v_mul(delta_, p_xi);

    
    Params theta;
}

