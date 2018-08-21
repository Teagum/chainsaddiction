#include "fwbw.h"
#include "stdio.h"

typedef struct {
    Vector *_lambda;
    Matrix *_gamma;
    Vector *delta;
} Params;


void fwbw(Vector *x, size_t m, size_t n, Vector *_lambda, Matrix *_gamma, Vector *_delta)
{
	
	Scalar sai = 0;		/* sum alpha_i */
	Scalar lsf = 0;		/* log scale factor */

	Vector *prob_i = NewEmptyVector(_lambda->n);
	Vector *alpha_i = NewEmptyVector(_lambda->n);

	Matrix *alpha = NewEmptyMatrix(n, m);
	/* Matrix *beta  = NewMatrix(n, m, 0L); */

	
	ppmf(_lambda, x->data[0], prob_i);
	printf("prob_i:\n");
	print_vector(prob_i);

	alpha_i = v_v_mul(_delta, prob_i);
	printf("alpha_i:\n");
	print_vector(alpha_i);	

	sai = v_sum(alpha_i);
	lsf = log1pl(sai);
	printf("sai: %Lf, lsf %Lf", sai, lsf);

	/*
	v_s_DIV(alpha_i, sai); 
	v_s_ADD(alpha_i, lsf);
	
	m_set_row(alpha, 0, alpha_i);


	print_vector(alpha_i);
	*/
	/* Params theta;
	 * return theta;
	 */
}




