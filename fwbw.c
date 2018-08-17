#include <math.h>
#include "linalg.h"


typedef struct {
    Vector *lambda_;
    Matrix *gamma_;
    Vector *delta;
} Params;


Params *fwbw(Vector *x, size_t m, size_t n, Vector *lambda_, Matrix *gamma_, Vector *delta_)
{
    
    Matrix *alpha = NewEmptyMatrix(n, m):
    Matrix *beta  = NewEmptyMatrix(n, m):

    Vector *a_t =  v_v_mul(delta_, p_xi);


    
    Params theta;
}

