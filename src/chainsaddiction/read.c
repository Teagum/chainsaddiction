#include "read.h"


enum io_error {
    IO_READ_ERROR,
    IO_FORMAT_ERROR
};


inline FILE *
Ca_OpenFile (const char *path, const char *mode)
{
    return fopen (path, mode);
}


inline int
Ca_CloseFile (FILE *file)
{
    return fclose (file);
}


size_t
Ca_ReadDataFile (FILE *stream, size_t n_lines, scalar *target)
{
    int ferr = 0;
    size_t lcnt = 0;

    while (lcnt < n_lines)
    {
        ferr = fscanf (stream, RSF, target);
        if (ferr == EOF)
        {
            if (ferror (stream))
            {
                Ca_ErrMsg ("ERROR: failed reading from stream.");
                break;
            }
            else if (feof (stream))
            {
                break;
            }
        }
        else if (ferr != 1)
        {
            Ca_ErrMsg ("ERROR: Content of line does not match format.");
        }
        lcnt++;
        target++;
    }
    return lcnt;
}


void
Ca_CountLines (FILE *stream, size_t *line_cnt)
{
    int fp_err = 1;
    int chr = 0;
    fpos_t fp_current;

    if (stream == NULL)
    {
        PERISH ("File pointer points to NULL");
    }

    fp_err = fgetpos (stream, &fp_current);
    if (fp_err)
    {
        PERISH ("Could not get file position");
    }

    rewind (stream);

    while (true)
    {
        chr = getc (stream);
        if (chr == EOF) break;

        if (chr == '\n')
        {
            ++*line_cnt;
        }
    }

    fp_err = fsetpos (stream, &fp_current);
    if (fp_err)
    {
        PERISH ("Could not set file position");
    }
}


int
Ca_ReadSectionHeader (
    FILE *stream,
    const char *header)
{
    char  buff[HEADER_SIZE_MAX];
    char *status   = NULL;
    int   is_equal = 0;

    if (header == NULL) {
        fprintf (stderr, "Invalid header name.\n");
        RETURN_ERROR;
    }
    const size_t len = strlen (header);

    if (stream == NULL) {
        fprintf (stderr, "Invalid file stream.\n");
        RETURN_ERROR;
    }

    status = fgets (buff, len+1, stream);
    if (status == NULL) {
        fprintf (stderr, "Could not read section header ``%s''.\n", header);
        RETURN_ERROR;
    }

    is_equal = strcmp (header, buff);
    if (is_equal != 0) {
        fprintf (stderr, "Header does not match.\n");
        RETURN_ERROR;
    }
    RETURN_SUCCESS;
}


int
Ca_ReadSectionData (
    FILE *stream,
    const size_t n_elem,
    scalar *buff)
{
#ifdef _NO_LD_MATH
    const char fmt[] = "%lf ";
#else
    const char fmt[] = "%Lf ";
#endif
    int status = 0;

    if (stream == NULL) {
        fprintf (stderr, "Invalid file stream.\n");
        RETURN_ERROR;
    }

    if (n_elem == 0 || n_elem > M_STATES_MAX) {
        fprintf (stderr, "Invalid number of data elements.\n");
        RETURN_ERROR;
    }

    if (buff == NULL) {
        fprintf (stderr, "Invalid output buffer.\n");
        RETURN_ERROR;
    }

    for (size_t i = 0; i < n_elem; i++) {
        status = fscanf (stream, fmt, buff++);
        if (status != 1) {
            fprintf (stderr, "Could not read element %zu of %zu.\n", i, n_elem);
            RETURN_ERROR;
        }
    }
    RETURN_SUCCESS;
}
