#include <stdio.h>
#include "linalg.h"

int main()
{
    Vector *a = NewVector(5, 1);
    print_vector(a);

    Vector *b = v_d_div(a, 0);
    if (b == NULL)
        fprintf(stderr, "AHHHH, ZERO DIVISION!");
    print_vector(a);
    print_vector(b);

}

