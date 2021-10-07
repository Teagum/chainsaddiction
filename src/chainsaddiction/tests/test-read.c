#include "test-read.h"


bool
test_Ca_ReadDataFile_full_file (void)
{
    const char path[] = "../../../tests/data/earthquakes/earthquakes";

    bool   err     = true;
    size_t n_lines = 0;
    size_t r_lines = 0;

    scalar *data = NULL;
    FILE   *file = NULL;

    file = fopen (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could not open data file.");
        err = true;
        goto cleanup;
    }

    Ca_CountLines (file, &n_lines);
    data = malloc (sizeof (scalar) * n_lines);
    if (data == NULL)
    {
        Ca_ErrMsg ("Could not allocate buffer.");
        err = true;
        goto cleanup;
    }
    r_lines = Ca_ReadDataFile (file, n_lines, data);
    err = false;

cleanup:
    fclose (file);
    free (data);
    return err ? UT_FAILURE : UT_SUCCESS;
}


bool
test_Ca_ReadDataFile_n_lines (void)
{
    cnt r_lines = 0;
    cnt n_lines = 0;
    scalar *data = NULL;
    FILE *file = NULL;
    enum { n_runs = 100 };

    for (cnt i=0; i < n_runs; i++)
    {
        n_lines = rand () % N_EQ;
        file = Ca_OpenFile ("tests/data/earthquakes", "r");
        S_MALLOC (data, n_lines);
        r_lines = Ca_ReadDataFile (file, n_lines, data);
        Ca_CloseFile (file);
        free (data);

        if (r_lines != n_lines) {
            return true;
        }
    }
    return false;
}


bool
test_Ca_CountLines_earthquakes (void)
{
    bool err = true;
    cnt n_lines = 0;

    FILE *file = Ca_OpenFile ("tests/data/earthquakes", "r");
    Ca_CountLines (file, &n_lines);

    if (n_lines == N_EQ)
    {
        err = false;
    }
    else
    {
        err = true;
    }

    Ca_CloseFile (file);
    return err;
}


bool
test_Ca_CountLines_empty (void)
{
    bool err = true;
    cnt n_lines = 0;

    FILE *file = Ca_OpenFile ("tests/data/empty", "r");
    Ca_CountLines (file, &n_lines);

    if (n_lines == 0)
    {
        err = false;
    }
    else
    {
        err = true;
    }

    Ca_CloseFile (file);
    return err;
}


bool
test_Ca_CountLines_wrong_format (void)
{
    bool err = true;
    cnt n_lines = 0;

    FILE *file = Ca_OpenFile ("tests/data/wrong_format", "r");
    Ca_CountLines (file, &n_lines);

    if (n_lines == 3)
    {
        err = false;
    }
    else
    {
        err = true;
    }

    Ca_CloseFile (file);
    return err;
}
