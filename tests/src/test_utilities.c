#include "test_utilities.h"

DataSet *ReadFromStdin(void)
{
    size_t N       = 100;
    size_t row_cnt = 0;
    size_t max_len = 50;

    char buffer[max_len];
    void *mem_buffer = NULL;
    char *e;

    DataSet *X = malloc (sizeof (*X));
    if (X == NULL) return NULL;

    X->data = malloc (N * sizeof (*X->data));
    if (X->data == NULL) goto error;

    while (fgets (buffer, max_len, stdin))
    {
        if (!(row_cnt < N))
        {
            N += 50;
            mem_buffer= realloc (X->data, N);
            if (mem_buffer == NULL) goto error;
            X->data = mem_buffer;
            mem_buffer = NULL;
        }

        X->data[row_cnt] = strtol (buffer, &e, 10);
        fprintf (stdout, "%zu : %ld\n", row_cnt, X->data[row_cnt]);

        if (errno != 0)
            fprintf(stderr, "Error reading line %zu. Skipping it.\n", row_cnt);

        row_cnt++;
    }
    printf("\n\n");
    N = row_cnt;
    //X->data = realloc (X->data, N);
    X->size = N;

    return X;

error:
    free_DataSet (X);
    return NULL;    
}

void free_DataSet(DataSet *X)
{
    free (X->data);
    free (X);
}
