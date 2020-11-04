#include "utilities.h"

#define max_len 60
#define N 10

scalar *
_alloc_block (
    const size_t n_elem)
{
    scalar *block = malloc (n_elem * sizeof *block);
    CHECK_ALLOC_FAIL (block, "Could not allocate block.");
    return block;
}


scalar *
_alloc_block_fill (
    const size_t n_elem,
    const scalar val)
{
    scalar *block = NULL;
    if (val == 0.0L)
    {
        block = calloc (n_elem, sizeof *block);
        CHECK_ALLOC_FAIL (block, "Could not allocate block.");
        return block;
    }
    block = _alloc_block (n_elem);
    for (size_t i = 0; i < n_elem; i++)
    {
        block[i] = val;
    }
    return block;
}


DataSet *read_dataset ()
{
    char buffer[max_len];
    size_t row_cnt = 0;

    DataSet *X = malloc (sizeof *X);
    if (X == NULL) goto exit_point;

    X->data = NULL;
    X = realloc_dataset (X, N);
    if (X == NULL) goto exit_point;

    while (fgets (buffer, max_len, stdin))
    {
        if (!(row_cnt < N))
        {
            X = realloc_dataset (X, X->size+20);
            if (X == NULL) goto exit_point;
        }
        X->data[row_cnt] = strtol (buffer, NULL, 10);

        if (errno != 0)
            fprintf(stderr, "Error reading line %zu. Skipping it.\n", row_cnt);

        row_cnt++;
    }

    X = realloc_dataset (X, row_cnt);

exit_point:
    return X;
}

void free_dataset(DataSet *X)
{
    free (X->data);
    free (X);
}


DataSet *realloc_dataset(DataSet *X, size_t new_size)
{
    void *mem_buffer = realloc (X->data, new_size * sizeof (*X->data));
    if (mem_buffer == NULL)
    {
        fprintf(stderr, "Error in realloc\n");
        free_dataset (X);
        return NULL;
    }

    X->size = new_size;
    X->data = mem_buffer;

    return X;
}
