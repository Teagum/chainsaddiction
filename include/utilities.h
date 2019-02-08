#ifndef utilities_h
#define utilities_h

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>


typedef struct {
    long   *data;
    size_t size;
} DataSet;


/** Read newline-seperated values from the standard input. 
 * This fucntion allows the programm to read files provided 
 * via output redirection by the command line. The values in
 * the file need to be seperated by newline ("\n"). 
 *
 * This function returns NULL on failure.
 */
DataSet *read_dataset();


/** Frees memory pointed to by an allocated DataSet* pointer.
 */
void free_dataset(DataSet *X);


/** Change the size of a DataSet. 
 * Changes the memory area pointed to by the `data` member of a DataSet.
 * The `data` members's size is changed according to `new_size`. `size` is
 * updated, too. On failure `data` AND the DataSet itselfs are freed, `size`
 * is not updated and NULL is returned.
 */
DataSet *realloc_dataset(DataSet *X, size_t new_size);

#endif    /* utilities_h */
