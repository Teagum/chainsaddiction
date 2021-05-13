#include "test_fwbw.h"


int
main (void)
{
    SETUP;

    RUN_TEST (test_log_forward);
    /*j
    scalar *lpp = alloc_block (m_states * data->size);
    scalar *alpha = alloc_block (m_states * data->size);
    scalar *beta = alloc_block (m_states * data->size);

    v_poisson_logpmf (data->data, data->size, lambda, m_states, lpp);
    log_forward_backward (lpp, gamma, delta, m_states, data->size, alpha, beta);

    for (size_t i = 0; i < data->size; i++)
    {
        fprintf (stdout, "[%4zu]\t", i);
        for (size_t j = 0; j < m_states; j++)
        {
            fprintf (stdout, "%10.5Lf\t", alpha[i*m_states+j]);
        }
        fprintf (stdout, "|\t");
        for (size_t j = 0; j < m_states; j++)
        {
            fprintf (stdout, "%10.5Lf\t", beta[i*m_states+j]);
        }
        fprintf (stdout, "\n");
    }
    */
    EVALUATE;
}

bool
test_log_forward (void)
{
    const size_t m_states = 3;
    const scalar lambda[] = {10, 20, 30};
    const scalar gamma[] = {.8, .1, .1,
                            .1, .8, .1,
                            .1, .1, .8};
    const scalar delta[] = {0.333333, 0.333333, .333333};

    vi_log (gamma, m_states * m_states);
    vi_log (delta, m_states);

    puts ("\n");
    for (size_t i = 0; i < m_states; i++) {
        for (size_t j = 0; j < m_states; j++) {
            printf ("%10.5Lf", gamma[i*m_states+j]);
        }
    }
    return true;
}
