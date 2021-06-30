#ifndef pois_probs_h
#define pois_probs_h

#include <stdlib.h>
#include <string.h>
#include "../libma.h"
#include "../restrict.h"
#include "../scalar.h"


/** Computation buffer for HMM estimation.
 *
 * Each field is a pointer to continuous memory with
 * space for n_obs * m_states values.
 */
typedef struct {
    size_t n_obs;
    size_t m_states;
    scalar *lsdp;       /**< Log of the state dependent probabilities   */
    scalar *lalpha;     /**< Log forward probabilities                  */
    scalar *lbeta;      /**< Log backward probabilities                 */
    scalar *lcxpt;      /**< Log conditional expectations               */
} PoisProbs;


#define PoisProbs_Delete(probs) do {    \
    free (probs->lsdp);                 \
    free (probs->lalpha);               \
    free (probs->lbeta);                \
    free (probs->lcexpt);               \
    free (probs)                        \
    probs = NULL;                       \
} while (false)


PoisProbs *
PoisProbs_New (
    const scalar n_obs,
    const scalar m_states);


#endif  /* pois_probs_h */
