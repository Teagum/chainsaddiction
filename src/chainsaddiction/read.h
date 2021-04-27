#ifndef read_h
#define read_h

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "scalar.h"

#define cnt unsigned int

#define PERISH(msg) do {        \
    perror (msg);               \
    exit (EXIT_FAILURE);        \
} while (1)


/** Open a file.
 *
 * Try to open path and check for possible errors.
 *
 * \param[in]   path    Path to file.
 * \param[in]   mode    File mode
 *
 * \return  Pointer to open file stream.
 */
extern FILE *
Ca_OpenFile (const char *path, const char *mode);


/**Close a file and check for errors.
  *
  * Try to close `file' and check for possible errors.
  *
  *\param[in]   file    Open file stream.
  *
*/
extern void
Ca_CloseFile (FILE *file);


/** Read new line separated files
  *
  *\param[in]   file    Open file stream.
  *\param[out]  target  Pointer to allocated memory.
  *
  *\return  Number of read lines.
  */
extern cnt
Ca_ReadDataFile (FILE *stream, cnt n_lines, scalar *target);


/** Count to number of lines in a file.
 *
 * Count the number of lines in a file until EOF is reached or an error
 * occures.
 *
 * \params[in]  file    Open file strea.
 * \params[out] cnt     Number of lines.
 *
 * \return  Error indicator.
 */
extern void
Ca_CountLines (FILE *file, cnt *line_cnt);


#endif	/* read_h */
