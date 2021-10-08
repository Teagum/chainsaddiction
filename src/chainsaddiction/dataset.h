#ifndef dataset_h
#define dataset_h

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "chainsaddiction.h"
#include "libma.h"
#include "read.h"


#define OUT_OF_BOUNDS_ERR_MSG   \
    "ERROR: Index %zu out of bounds for dimension of size %zu.\n"


typedef struct {
    scalar *data;
    size_t size;
    bool err;
} DataSet;


/** Deallocate struct DataSet.
 */
#define DataSet_Delete(this)    \
if (this != NULL)               \
{                               \
    MA_FREE (this->data);       \
    MA_FREE (this);             \
}


/** Create a new DataSet.
 *
 * \return  Pointer to DataSet.
 */
DataSet *
DataSet_NewEmpty (void);


/** Create a new DataSet with `n_elem` entries.
 *
 * Each entry is initialized with zero.
 *
 * \param n_elem    Number of elements in data set.
 *
 * \return  Pointer to DataSet.
 */
DataSet *
DataSet_New (const size_t n_elem);


/** Creat a new DataSet from a data file.
 *
 *\param path   Path to data file.
 */
DataSet *
DataSet_NewFromFile (const char *path);


/** Set a single element of DataSet.
 *
 * Perform bounds checks and set the element.
 *
 * Set DataSet.err = false on success.
 * Set DataSet.err = true on failure.
 *
 *\param[in] this    Pointer to dataset.
 *\param[in] idx    Element index.
 *\param[in] val    Value.
 */
extern void
DataSet_SetValue (DataSet *restrict this, size_t idx, scalar val);


/** Get a single element of a DataSet.
 *
 * Perform bounds checks then read the element and copy
 * it to the addrees of `out`.
 *
 * Set DataSet.err = false on success.
 * Set DataSet.err = true on failure.
 *
 *\param[in]  this    Pointer to dataset.
 *\param[in]  idx    Element index.
 *\param[out] val    Write the value to the adress of `val`.
 */
extern void
DataSet_GetValue (DataSet *restrict this, size_t idx, scalar *out);


/** Print the contents of a DataSet.
 *
 */
extern void
DataSet_Print (DataSet *this);


#endif    /* dataset_h */
