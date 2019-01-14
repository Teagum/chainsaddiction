#include <stdio.h>
#include "test_utilities.h"

int main(void)
{
    DataSet *X = ReadFromStdin();

    printf("%zu\n", X->size);
    for (size_t i = 0; i < X->size; i++)
        fprintf (stdout, "%zu : %ld\n", i, X->data[i]);

    free_DataSet(X);
    return 0;
}
