#ifndef EM_H
#define EM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fwbw.h"
#include "scalar.h"
#include "hmm.h"

int poisson_expectation_maximization(
		const	long		*x,
		const	size_t		n,
				PoissonHMM	*ph);

#endif	/* EM_H */
