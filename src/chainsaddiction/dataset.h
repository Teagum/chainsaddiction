#ifndef dataset_h
#define dataset_h

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include "restrict.h"
#include "scalar.h"
#include "libma.h"

#define DATASET_INIT_SIZE 10
#define DATASET_MEM_INC 100

#define OUT_OF_BOUNDS_ERR_MSG     \
    "ERROR: Index %zu out of bounds for dimension of size %zu.\n"


typedef struct {
    scalar *data;
    size_t size;
    bool err;
} DataSet;


/** Deallocate struct DataSet.
 */
#define CA_FREE_DATASET(pds)            \
do {                                    \
    MA_FREE (pds->data);                \
    MA_FREE (pds);                      \
} while (0)


/** Create a new DataSet
 */
DataSet *
Ca_NewDataSet (void);


/** Set a single element of DataSet.
 *
 * Perform bounds checks and set the element.
 *
 * Set DataSet.err = false on success.
 * Set DataSet.err = true on failure.
 *
 *\param[in] pds    Pointer to dataset.
 *\param[in] idx    Element index.
 *\param[in] val    Value.
 */
extern void
ds_set (DataSet *restrict pds, size_t idx, scalar val);


/** Get a single element of a DataSet.
 *
 * Perform bounds checks then read the element and copy
 * it to the addrees of `out`.
 *
 * Set DataSet.err = false on success.
 * Set DataSet.err = true on failure.
 *
 *\param[in]  pds    Pointer to dataset.
 *\param[in]  idx    Element index.
 *\param[out] val    Write the value to the adress of `val`.
 */
extern void
ds_get (DataSet *restrict pds, size_t idx, scalar *out);


#endif    /* dataset_h */
