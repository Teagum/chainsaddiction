#include "test_read.h"


unsigned short N_ERRORS = 0;

int main (void)
{
    SETUP;

    RUN_TEST (test_Ca_ReadDataFile_full_file);
    RUN_TEST (test_Ca_ReadDataFile_n_lines);
    RUN_TEST (test_Ca_CountLines_earthquakes);
    /*
    FEEDBACK (test_Ca_CountLines_empty);
    FEEDBACK (test_Ca_CountLines_wrong_format);
    */

    EVALUATE;
}


bool
test_Ca_ReadDataFile_full_file (void)
{
    bool fail = true;
    cnt n_lines = 0;
    cnt r_lines = 0;
    scalar *data = NULL;
    FILE *file = NULL;

    file = Ca_OpenFile ("tests/data/earthquakes", "r");
    Ca_CountLines (file, &n_lines);
    S_MALLOC (data, n_lines);
    r_lines = Ca_ReadDataFile (file, n_lines, data);
    Ca_CloseFile (file);
    free (data);

    file = Ca_OpenFile ("tests/data/centroids", "r");
    Ca_CountLines (file, &n_lines);
    S_MALLOC (data, n_lines);
    r_lines = Ca_ReadDataFile (file, n_lines, data);
    Ca_CloseFile (file);
    free (data);

    fail = false;
    return fail;
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
