#ifndef EM_H
#define EM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fwbw.h"
#include "scalar.h"
#include "hmm.h"
#include "utilities.h"

int PoisHmm_EM (const DataSet *restrict x,
                      PoisHmm *restrict ph);

#endif  /* EM_H */
