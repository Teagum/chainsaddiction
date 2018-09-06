#include <stdio.h>
#include "matrix.h"
#include "stats.h"

int main(void)
{
	const size_t m 	= 3;
	scalar	lam[m]	= { 10, 20, 30 };
	scalar	out[m]	= { };
	long	x		= 35L;

	ppmf(lam, m, x, out);

	for (size_t i = 0; i < m; i++)
	{
		printf("%Lf\t", out[i]);
	}
	printf("\n");
	return 0;
}
