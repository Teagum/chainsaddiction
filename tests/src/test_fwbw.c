#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fwbw.h"
#include "test_utilities.h"

int main(int argc, char *argv[])
{
    size_t m = 3;
    size_t n = 50;
    long *x = malloc (n * sizeof (long));

    n = read_stdin(x, n);   

    scalar  l[] = { 10L, 20L, 30L };
    scalar  g[] = { .8, .1, .1,
                    .1, .8, .1,
                    .1, .1, .8 };
    scalar  d[] = { 1.0/3.0, 1.0/3.0, 1.0/3.0 };

    scalar *alpha = malloc (n * m * sizeof (scalar));
    scalar *beta  = malloc (n * m * sizeof (scalar));
    scalar *probs = malloc (n * m * sizeof (scalar));
    if (alpha == NULL || beta == NULL || probs == NULL)
    {
        free (alpha); free (beta); free (probs); return 0;
    }

    scalar *data = NULL;

    for (size_t i = 0; i < n; i++)
        printf("%zu: %ld\n", i, x[i]);

    return 0;
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
    free(x);
    return 0;
}
