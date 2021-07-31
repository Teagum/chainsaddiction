#include "read.h"


enum io_error {
    IO_READ_ERROR,
    IO_FORMAT_ERROR
};


inline FILE *
Ca_OpenFile (const char *path, const char *mode)
{
    FILE *file = fopen (path, mode);
    if (file == NULL)
    {
        perror ("ERROR");
        exit (EXIT_FAILURE);
    }
    return file;
}


inline void
Ca_CloseFile (FILE *file)
{
    if (fclose (file) == EOF)
    {
        fputs ("ERROR: Failed to close file.", stderr);
        exit (EXIT_FAILURE);
    }
}


cnt
Ca_ReadDataFile (FILE *stream, cnt n_lines, scalar *target)
{
    int ferr = 0;
    cnt line_cnt = 0;

    while (line_cnt < n_lines)
    {
        ferr = fscanf (stream, RSF, target);
        if (ferr == EOF)
        {
            if (ferror (stream))
            {
                PERISH ("ERROR: failed reading from stream.");
            }
            else if (feof (stream))
            {
                break;
            }
        }
        else if (ferr != 1)
        {
            PERISH ("ERROR: Content of line does not match format.");
        }
        line_cnt++;
        target++;
    }
    return line_cnt;
}


void
Ca_CountLines (FILE *file, cnt *line_cnt)
{
    int fp_err = 1;
    int chr = 0;
    fpos_t fp_start = 0;
    fpos_t fp_current = 0;

    if (file == NULL)
    {
        PERISH ("File pointer points to NULL");
    }

    fp_err = fgetpos (file, &fp_current);
    if (fp_err)
    {
        PERISH ("Could not get file position");
    }

    if (fp_current != fp_start)
    {
        fp_err = fsetpos (file, &fp_start);
        if (fp_err)
        {
            PERISH ("Could not set file position");
        }
    }

    while (true)
    {
        chr = getc (file);
        if (chr == EOF) break;

        if (chr == '\n')
        {
            ++*line_cnt;
        }
    }

    fp_err = fsetpos (file, &fp_current);
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
