#ifndef test_read_h
#define test_read_h

#include "unittest.h"
#include "chainsaddiction.h"
#include "read.h"
#include "libvmath.h"


#define N_EQ 107


bool
test_Ca_ReadDataFile_full_file (void);

bool
test_Ca_ReadDataFile_n_lines (void);

bool
test_Ca_CountLines_earthquakes (void);

bool
test_Ca_CountLines_empty (void);

bool
test_Ca_CountLines_wrong_format (void);


#endif	/* test_read_h */
