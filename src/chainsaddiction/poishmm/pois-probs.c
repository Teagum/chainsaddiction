#include "pois-probs.h"


PoisProbs *
PoisProbs_New (
    const size_t n_obs,
    const size_t m_states)
{
    size_t n_elem = n_obs * m_states;
    if (n_obs == 0)
    {
        fprintf (stderr, "`n_obs' must be greater than 0.\n");
        return NULL;
    }
    if (m_states == 0)
    {
        fprintf (stderr, "`m_states' must be greater than 0.\n");
        return NULL;
    }
    if (n_elem / n_obs != m_states)
    {
        fprintf (stderr, "Integer overflow detected.");
        return NULL;
    }

    PoisProbs *probs = malloc (sizeof *probs);
    if (probs == NULL)
    {
        fprintf (stderr, "Could not allocate memory for PoisProbs object.\n");
        return NULL;
    }
    probs->lsdp     = MA_SCALAR_ZEROS (n_elem);
    probs->lalpha   = MA_SCALAR_ZEROS (n_elem);
    probs->lbeta    = MA_SCALAR_ZEROS (n_elem);
    probs->lcsp     = MA_SCALAR_ZEROS (n_elem);
    probs->n_obs    = n_obs;
    probs->m_states = m_states;

    return probs;
}
