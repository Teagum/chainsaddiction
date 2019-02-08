#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fwbw.h"
#include "utilities.h"
#include "hmm.h"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf (stdout, "Usage: test_fwbw -[abpn] m param_file < dataset\n");
        return 0;
    }

    size_t m = atoi (argv[2]);
    scalar *output = NULL;

    PoisParams *params = PoisHmm_ParamsFromFile(argv[3]);
    if (params == NULL)
    {
        fprintf (stderr, "Error reading parameter file.\n");
        return  1;
    }

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
        fprintf (stderr, "Allocation error.\n");
        free (alpha); free (beta); free (probs); return 1;
    }

    if (strcmp(argv[1], "-a") == 0) output = alpha;
    else if (strcmp(argv[1], "-b") == 0) output = beta;
    else if (strcmp(argv[1], "-p") == 0) output = probs;
    else output = alpha;

    PoisHmm_FwBw (X->data, X->size, m, params, alpha, beta, probs);

    for (size_t i = 0; i < X->size; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            printf("%40.30Lf", output[i*m+j] );
        }
        printf("\n");
    }

    free (alpha);
    free (beta);
    free (probs);
    free_dataset (X);
    PoisHmm_FreeParams (params);

    return 0;
}
