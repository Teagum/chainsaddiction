#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fwbw.h"
#include "utilities.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf (stdout, "Usage: test_fwbw -[abp] m < dataset\n");
        return 0;
    }

    size_t m = atoi (argv[2]);
    scalar *output = NULL;


    scalar  l[] = { 10L, 20L, 30L };
    scalar  g[] = { .8, .1, .1,
                    .1, .8, .1,
                    .1, .1, .8 };
    scalar  d[] = { 1.0/3.0, 1.0/3.0, 1.0/3.0 };

    DataSet *X = read_dataset();
    if (X == NULL)
    {
        fprintf (stderr, "Error while reading dataset.\n");
        return 1;
    }

    scalar *alpha = malloc (X->size * m * sizeof (scalar));
    scalar *beta  = malloc (X->size * m * sizeof (scalar));
    scalar *probs = malloc (X->size * m * sizeof (scalar));
    if (alpha == NULL || beta == NULL || probs == NULL)
    {
        free (alpha); free (beta); free (probs); return 0;
    }

    if (strcmp(argv[1], "-a") == 0) output = alpha;
    else if (strcmp(argv[1], "-b") == 0) output = beta;
    else if (strcmp(argv[1], "-p") == 0) output = probs;
    else output = alpha;

    log_poisson_forward_backward(X->data, X->size, m, l, g, d, alpha, beta, probs);


    for (size_t i = 0; i < X->size; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            printf("%30.20Lf", output[i*m+j] );
        }
        printf("\n");
    }

    free (alpha);
    free (beta);
    free (probs);
    free_dataset (X);
    
    return 0;
}
