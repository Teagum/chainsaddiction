#include <stdio.h>
#include "fwbw.h"
#include "scalar.h"
#include "utilities.h"

void log_v_inplace (scalar *arr, size_t n_elem);

int main (void)
{
    const size_t m_states = 3;
    scalar lambda[] = {10, 20, 30};
    scalar gamma[] = {.8, .1, .1,
                      .1, .8, .1,
                      .1, .1, .8};
    scalar delta[] = {0.333333, 0.333333, .333333};
    log_v_inplace (gamma, m_states * m_states);
    log_v_inplace (delta, m_states);

    DataSet *data = read_dataset ();

    scalar *lpp = _alloc_block (m_states * data->size);
    scalar *alpha = _alloc_block (m_states * data->size);
    scalar *beta = _alloc_block (m_states * data->size);

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
    return EXIT_SUCCESS;
}

void log_v_inplace (scalar *arr, size_t n_elem)
{
    for (size_t i = 0; i < n_elem; i++)
    {
        arr[i] = logl (arr[i]);
    }
}
