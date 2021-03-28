#include <stdio.h>
#include "dataset.h"

int main (void)
{
    DataSet *inp = Ca_NewDataSet ();

    /*
    for (size_t i=-22; i<17; i++)
    {
        scalar x;
        bool status = ds_get (inp, i, &x);
        if (status)
            printf ("%zu:\t%Lf\n", i, x);
    }

    */

    scalar x = 0;
    ds_get (inp, -3, &x);
    printf ("Status: %d\tx: %Lf\n", inp->err, x);
    if (inp->err)
        puts ("Computation failed.\n");
    return 0;
}
