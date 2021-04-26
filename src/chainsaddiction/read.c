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
        ferr = fscanf (stream, "%40Lf", target);
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
