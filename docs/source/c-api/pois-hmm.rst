.. default-domain:: c

HMM Layer
===============================================================================

Data structure
-------------------------------------------------------------------------------

.. struct:: PoisHmm
    
    The :struct:`PoisHmm` structure encapsulates all data of
    Poisson-distributed HMM.

    .. code-block:: c
        
        typedef struct PoisHmm {
            size_t n_obs;
            size_t m_states;
            size_t n_iter;
            size_t max_iter;
            scalar tol;
            scalar aic;
            scalar bic;
            scalar llh;
            PoisParams *init;
            PoisParams *params;
            PoisProbs *probs;
        } PoisHmm;

    .. var:: size_t n_obs

        An integral number greater than zero that specifies the number of
        states in an HMM.

    .. member:: size_t m_states

        An integral number that specifies the number of observations, that is,
        the length of the dataset.

    .. member:: size_t n_iter

        An integral number specifying the number of the iterations it took the
        EM Algorithm to converge. If EM did not converge, this is equal to
        :var:`max_iter`.

    .. member:: size_t max_iter

        An positive interal number specifiying the maximum number of iterations of
        the EM algorithm.

    .. member:: scalar tol

        An floating point number specifying the update tolerance of the EM
        algorithm. The update of a previous M-step is only applied the update
        score of greater than :var:`tol`. 

    .. member:: scalar aic
    .. member:: scalar bic
    .. member:: scalar llh


Functions
-------------------------------------------------------------------------------

.. function:: PoisHmm *PoisHmm_New (  \
   const size_t n_obs,                  \
   const size_t m_states)


.. function:: void PoisHmm_Init (         \
    PoisHmm *const restrict this,           \
    const scalar *const restrict lambda,    \
    const scalar *const restrict gamma,     \
    const scalar *const restrict delta)


.. function:: void PoisHmm_InitRandom (PoisHmm *const restrict this)


.. function:: void PoisHmm_LogLikelihood (PoisHmm *const restrict this)
