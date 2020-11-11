#ifndef utilities_h
#define utilities_h

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include "scalar.h"

#define OUTER_LOOP for (size_t i = 0; i < n_elem; i++)
#define INNER_LOOP for (size_t j = 0; j < n_elem; j++)

#define CHECK_ALLOC_FAIL(buffer, msg) \
do { \
    if (buffer == NULL) \
    { \
        fprintf (stderr, "%s\n", msg); \
        exit (1); \
    } \
} while (0)

typedef struct {
    unsigned long *data;
    size_t size;
} DataSet;


/** Allocate continuous memory block.
 *
 * Allocate a continuous block memory for long double values and check
 * for allocation error.
 *
 * @param n_elem - Numnber of block elements.
 */
scalar
*alloc_block (
    const size_t n_elem);


/** Allocate continuous memory block initialized with value.
 *
 * Allocate a continuous block memory for long double values, check
 * for allocation error and initialize each block element with val. 
 *
 * @param n_elem - Numnber of block elements.
 */
scalar
*alloc_block_fill (
    const size_t n_elem,
    const scalar val);


/** Read newline-seperated values from the standard input. 
 * This fucntion allows the programm to read files provided 
 * via output redirection by the command line. The values in
 * the file need to be seperated by newline ("\n"). 
 *
 * This function returns NULL on failure.
 */
DataSet *
read_dataset ();


/** Frees memory pointed to by an allocated DataSet* pointer.
 */
void free_dataset(DataSet *X);


/** Change the size of a DataSet. 
 * Changes the memory area pointed to by the `data` member of a DataSet.
 * The `data` members's size is changed according to `new_size`. `size` is
 * updated, too. On failure `data` AND the DataSet itselfs are freed, `size`
 * is not updated and NULL is returned.
 */
DataSet *
realloc_dataset (
    DataSet *X,
    size_t new_size);


#endif    /* utilities_h */
