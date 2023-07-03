#ifndef read_h
#define read_h

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "chainsaddiction.h"

#define RETURN_ERROR return 1
#define RETURN_SUCCESS return 0
#define HEADER_SIZE_MAX 100
#define M_STATES_MAX 50

#define PERISH(msg) do {        \
    perror (msg);               \
    exit (EXIT_FAILURE);        \
} while (1)


/** Open a file.
 *
 * \param[in]   path    Path to file
 * \param[in]   mode    File mode
 *
 * \return  Pointer to open file stream or NULL
 */
extern FILE *
Ca_OpenFile (const char *path, const char *mode);


/** Close an open file stream.
  *
  * \param[in]   file    Open file stream.
  *
  * \return  0 on suceess, `EOF` on failure.
*/
extern int
Ca_CloseFile (FILE *file);


/** Read numerical data from file.
 *
 * Numbers have to be separated by newline.
 *
 * \param[in]   file    Open file stream.
 * \param[out]  target  Pointer to allocated memory.
 *
 * \return  Number of read lines.
*/
extern size_t
Ca_ReadDataFile (FILE *stream, size_t n_lines, scalar *target);


/** Count to number of lines in a file.
 *
 * Count the number of lines in a file until EOF is reached or an error
 * occures.
 *
 * \params[in]  file     Open file stream.
 * \params[out] n_lines  Number of lines.
 *
 * \return  Error indicator.
 */
extern void
Ca_CountLines (FILE *stream, size_t *n_lines);


extern int
Ca_ReadSectionHeader (
    FILE *stream,
    const char *header);


extern int
Ca_ReadSectionData (
    FILE *stream,
    const size_t n_elem,
    scalar *buff);


#endif	/* read_h */
