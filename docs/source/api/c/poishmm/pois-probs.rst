.. default-domain:: c

Computation buffers
-------------------------------------------------------------------------------

.. struct:: PoisProbs

    The :struct:`PoisProbs` structure houses computation buffers that
    are shared between algorithms.

    .. code-block:: c

        typedef struct PoisProbs {
            size_t n_obs;
            size_t m_states;
            scalar *lsdp;
            scalar *lalpha;
            scalar *lbeta;
            scalar *lcxpt;
        } PoisProbs;


    .. member:: size_t PoisProbs.n_obs

        Number of observations in the input data set.

    .. member:: size_t PoisProbs.m_states

        Number of states in the HMM.

    .. member:: scalar *PoisProbs.lsdp
        
        Points to an object of :member:`n_obs` times :member:`m_states` elements
        that stores the logarithm of the state-dependend probabilities of the
        input data under the Poisson distribution for each state.

    .. member:: scalar *PoisProbs.lalpha

        Points to an object of :member:`n_obs` times :member:`m_states` elements
        that stores the logarithm of the forward probabilities of the input
        data under the model parameters.

    .. member:: scalar *PoisProbs.lbeta

        Points to an object of :member:`n_obs` times :member:`m_states` elements
        that stores the logarithm of the backward probabilities of the input
        data under the model parameters.

    .. member:: scalar *PoisProbs.lcsp

        Points to an object of :member:`n_obs` times :member:`m_states` elements
        that stores the logarithm of the conditional state probabilities.


Object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: PoisProbs * PoisProbs_New (const size_t n_obs, const size_t m_states)

    Allocate memory for a new :struct:`PoisProbs` structure.

.. macro:: PoisProbs_Delete (this)

    Delete a :struct:`PoisProbs` from memory.
