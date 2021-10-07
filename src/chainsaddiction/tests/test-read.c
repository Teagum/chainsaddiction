#include "test-read.h"


bool
test_Ca_ReadDataFile_full_file (void)
{
    const char path[] = "../../../tests/data/earthquakes/earthquakes";

    bool   err     = UT_FAILURE;
    size_t n_lines = 0;
    size_t r_lines = 0;

    scalar *data = NULL;
    FILE   *file = NULL;

    file = fopen (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could not open data file.");
        goto cleanup;
    }

    Ca_CountLines (file, &n_lines);
    data = malloc (sizeof (scalar) * n_lines);
    if (data == NULL)
    {
        Ca_ErrMsg ("Could not allocate buffer.");
        goto cleanup;
    }
    r_lines = Ca_ReadDataFile (file, n_lines, data);
    err = (r_lines == N_EQ) ? UT_SUCCESS : UT_FAILURE;

cleanup:
    fclose (file);
    free (data);
    return err;
}


bool
test_Ca_ReadDataFile_n_lines (void)
{
    const char path[] = "../../../tests/data/earthquakes/earthquakes";

    bool   err       = UT_FAILURE;
    size_t r_lines   = 0;
    size_t n_lines   = 0;
    size_t max_lines = 0;
    size_t min_lines = 1;

    scalar *data = NULL;
    FILE   *file = NULL;

    file = Ca_OpenFile (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could not open data file.");
        goto cleanup;
    }

    Ca_CountLines (file, &max_lines);
    n_lines = rnd_size (min_lines, max_lines);
    data = malloc (sizeof (scalar) * n_lines);
    if (data == NULL)
    {
        Ca_ErrMsg ("Could not allocate buffer.");
        goto cleanup;
    }

    r_lines = Ca_ReadDataFile (file, n_lines, data);
    err = (r_lines == n_lines) ? UT_SUCCESS : UT_FAILURE;

cleanup:
    Ca_CloseFile (file);
    free (data);
    return err;
}


bool
test_Ca_CountLines_earthquakes (void)
{
    const char path[] = "../../../tests/data/earthquakes/earthquakes";
    bool   err        = UT_FAILURE;
    size_t n_lines    = 0;
    FILE   *file      = NULL;

    file = Ca_OpenFile (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could not allocate buffer.");
        goto cleanup;
    }

    Ca_CountLines (file, &n_lines);
    err = (n_lines == N_EQ) ? UT_SUCCESS : UT_FAILURE;

cleanup:
    Ca_CloseFile (file);
    return err;
}


bool
test_Ca_CountLines_empty (void)
{
    const char   path[]  = "../../../tests/data/empty";
          bool   err     = UT_FAILURE;
          size_t n_lines = 0;
          FILE   *file   = NULL;

    file = Ca_OpenFile (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could open file.");
        goto cleanup;
    }

    Ca_CountLines (file, &n_lines);
    err = (n_lines == 0) ? UT_SUCCESS : UT_FAILURE;

cleanup:
    Ca_CloseFile (file);
    return err;
}


bool
test_Ca_CountLines_wrong_format (void)
{
    const char   path[]  = "../../../tests/data/wrong_format";
          bool   err     = UT_FAILURE;
          size_t n_lines = 0;
          FILE   *file   = NULL;

    file = Ca_OpenFile (path, "r");
    if (file == NULL)
    {
        Ca_ErrMsg ("Could open file.");
        goto cleanup;
    }

    Ca_CountLines (file, &n_lines);
    err = (n_lines == 3) ? UT_SUCCESS : UT_FAILURE;

cleanup:
    Ca_CloseFile (file);
    return err;
}
