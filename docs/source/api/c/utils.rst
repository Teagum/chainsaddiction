.. default-domain:: c

HMM utilities
===============================================================================

.. function:: scalar compute_aic (  \
      const size_t m_states,        \
      const scalar llh)

   Compute the `Akaike Information Criterion`_ from the number of states and
   the likelihood of a HMM.


.. function:: scalar compute_bic (  \
      const size_t n_obs,           \  
      const size_t m_states,        \
      const scalar llh)

   Compute the `Bayesian Information Criterion`_ from the number of
   obersavations, the number of states, and the likelihood of a HMM.


.. function:: scalar compute_log_likelihood (   \
      const size_t n_obs,                       \  
      const size_t m_states,                    \
      const scalar *const restrict lalpha)

   Compute the logarithm of the likelihood of a fitted HMM based on the forward
   probabilities.


.. function:: extern void log_csprobs (      \
      const size_t n_obs,                    \
      const size_t m_states,                 \
      const scalar llh,                      \
      const scalar *const restrict lalpha,   \
      const scalar *const restrict lbeta,    \
      scalar *const restrict lcsp)

   Compute the conditional state probabilities.

.. function:: extern int local_decoding (    \
      const size_t n_obs,                    \
      const size_t m_states,                 \
      const scalar *lcsp,                    \
      size_t *states)

   Compute the most likely hidden state under the HMM for each observation
   individually.

.. function:: extern int global_decoding (   \
      const size_t n_obs,                    \
      const size_t m_states,                 \
      const scalar *const restrict lgamma,   \
      const scalar *const restrict ldelta,   \
      const scalar *restrict lcsp,           \
      size_t *restrict states)

   Compute the most likely sequence of hidden states under the HMM using the
   `Viterbi algorithm`_.


.. _Akaike Information Criterion: https://en.wikipedia.org/wiki/Akaike_information_criterion

.. _Bayesian Information Criterion: https://en.wikipedia.org/wiki/Bayesian_information_criterion

.. _Viterbi algorithm: http://www.scholarpedia.org/article/Viterbi_algorithm
