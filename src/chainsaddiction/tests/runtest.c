#include "unittest.h"
#include "test-dataset.h"
#include "test-read.h"


int main (void)
{
    SETUP;

    RUN_TEST (test_DataSet_NewEmpty);
    RUN_TEST (test_DataSet_SetValue_error_on_idx_out_of_bounds);
    RUN_TEST (test_DataSet_SetValue);
    RUN_TEST (test_DataSet_GetValue_error_on_idx_out_of_bounds);
    RUN_TEST (test_DataSet_GetValue);

    RUN_TEST (test_Ca_ReadDataFile_full_file);
    RUN_TEST (test_Ca_ReadDataFile_n_lines);
    RUN_TEST (test_Ca_CountLines_earthquakes);
    RUN_TEST (test_Ca_CountLines_empty);
    RUN_TEST (test_Ca_CountLines_wrong_format);

    EVALUATE;
}
