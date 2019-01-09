#include <stdio.h>
#include <stdlib.h>

/* read values seperate by newline from 
 * a file as long integer. Return number of read values.
 */
size_t read_stdin (long *buffer, size_t n)
{
    size_t cnt = 0;
    size_t N   = n;

    while (!feof(stdin))
    {
        printf("%zu\n", cnt);
        if (cnt >= N-1)
        {
            puts("REALLOC");
            N += 100;
            buffer = realloc (buffer, N * sizeof (long));
        }

        if (fscanf(stdin, "%ld", &buffer[cnt++]) != 1)
            break;
    }

    N = cnt-1;
    buffer = realloc (buffer, N * sizeof (long));

    return N;
}
