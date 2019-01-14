#ifndef test_utilities_h
#define test_utilities_h

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

typedef struct {
    long   *data;
    size_t size;
} DataSet;


DataSet *ReadFromStdin();

void free_DataSet(DataSet *X);

#endif  /*test_utilities_h */
