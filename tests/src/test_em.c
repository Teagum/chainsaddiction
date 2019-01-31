#include <stdio.h>
#include "fwbw.h"
#include "em.h"
#include "hmm.h"
#include "utilities.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf (stdout, "Usage: test_em m < dataset\n");
    }

    size_t m = 3;
    size_t max_iter = 1000;
    scalar tol = 1e-6;

    DataSet *X = read_dataset();
    if (X == NULL)
    {
        fprintf (stderr, "Error while reading dataset.\n");
        return 1;
    }

    scalar ilambda[] = {10, 20, 30};
    scalar igamma[] = {.8, .1, .1,
                      .1, .8, .1,
                      .1, .1, .8};
    scalar idelta[] = {0.333333, 0.333333, .333333};

    PoisHmm* ph = PoisHmm_FromData (m, ilambda, igamma, idelta,
                                    max_iter, tol);

    
    int success = PoisHmm_EM (X, ph);
    if (success == 0)
    {
       fprintf (stderr, "EM failed.\n");
    }

    else
    {
        ph->aic = compute_aic (ph->nll, ph->m);
        ph->bic = compute_bic (ph->nll, ph->m, X->size);
        PoisHmm_PrintParams (ph->params, ph->m);

        fprintf (stdout, "NLL:\t%Lf\nAIC:\t%Lf\nBIC:\t%Lf\n", ph->nll, ph->aic, ph->bic);
        fprintf (stdout, "n_iter:\t%zu\n", ph->n_iter);
    } 

    free_dataset (X);
    PoisHmm_DeleteHmm (ph);
    return 0;
}
