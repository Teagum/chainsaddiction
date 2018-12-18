#include <stdio.h>
#include <string.h>
#include "fwbw.h"

int main(int argc, char *argv[])
{
	const size_t	m	=  3;
	const size_t    n	= 107;

	long	x[]	= {	13, 14,  8, 10, 16, 26, 32, 27, 18, 32,
						36, 24, 22, 23, 22, 18, 25, 21, 21, 14,
						 8, 11, 14, 23, 18, 17, 19, 20, 22, 19,
						13, 26, 13, 14, 22, 24, 21, 22, 26, 21,
						23, 24, 27, 41, 31, 27, 35, 26, 28, 36,
						39, 21, 17, 22, 17, 19, 15, 34, 10, 15,
						22, 18, 15, 20, 15, 22, 19, 16, 30, 27,
						29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
						18, 14, 10, 15,  8, 15,  6, 11,  8,  7,
						18, 16, 13, 12, 13, 20, 15, 16, 12, 18,
						15, 16, 13, 15, 16, 11, 11};

	scalar	l[]	= { 10L, 20L, 30L };
	scalar	g[]	= { .8, .1, .1,
						.1, .8, .1,
						.1, .1, .8 };
	scalar	d[]	= { 1./3., 1./3., 1./3. };

	scalar	*alpha	= malloc (n*m*sizeof(scalar));
	if (alpha == NULL) return 0;

	scalar	*beta	= malloc (n*m*sizeof(scalar));
	if (beta == NULL) { free(alpha); return 0; }

	scalar	*probs	= malloc (n*m*sizeof(scalar));
	if (probs == NULL) { free(alpha); free(beta); return 0; }

	scalar *data = NULL;

	log_poisson_forward_backward(x, n, m, l, g, d, alpha, beta, probs);

	if (argc < 2)
	{
		data = alpha;
	}
	else
	{
		if ( strcmp(argv[1], "-a") == 0 ) data = alpha;
		else if ( strcmp(argv[1], "-b") == 0 ) data = beta;
		else if ( strcmp(argv[1], "-p") == 0 ) data = probs;
		else 
		{
			printf("usage:\ttest_fwbw [-a | -b | -p]\n");
			return -1;
		}
	}

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			printf("%30.20Lf", data[i*m+j] );
		}
		printf("\n");
	}

	free(alpha);
	free(beta);
	free(probs);

	return 0;
}
