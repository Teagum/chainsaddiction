#ifndef test_rnd_h
#define test_rnd_h

#include <stdbool.h>
#include <math.h>
#include "config.h"
#include "assertations.h"
#include "rnd.h"


bool test__rnd_int (void);
bool test__v_rnd_int (void);

bool test__rnd_scalar (void);
bool test__v_rnd_scalar (void);

bool test__sample (void);
bool test__v_sample (void);


#endif  /* test_rnd_h */
