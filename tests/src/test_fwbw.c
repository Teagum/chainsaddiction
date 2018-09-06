#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "fwbw.h"

int main(int argc, char *argv[])
{
	const size_t	m	=  3;
	const size_t    n	= 10;

	long	x[n]	= { 49, 51, 44, 49, 52, 44, 56, 44, 49, 48 };
	scalar	l[m]	= { 10L, 20L, 30L };
	scalar	g[m*m]	= { .8, .1, .1,
						.1, .8, .1,
						.1, .1, .8 };
	scalar	d[m]	= { 1./3., 1./3., 1./3. };

	scalar	**alpha	= alloc_matrix(n, m);
	scalar	**beta	= alloc_matrix(n, m);
	scalar	**probs	= alloc_matrix(n, m);
	scalar	**data	= NULL;

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
			printf("%15.10Lf", *(*(data+i)+j) );
		}
		printf("\n");
	}

	free_matrix(alpha, n);
	free_matrix(beta, n);
	free_matrix(probs, n);

	return 0;
}
