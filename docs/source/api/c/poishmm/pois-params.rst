.. default-domain:: c


Parameters
-------------------------------------------------------------------------------

.. struct:: PoisParams

    .. code-block:: c

        typedef struct PoisParams {
            size_t m_states;
            scalar *lambda;
            scalar *gamma;
            scalar *delta;
        } PoisParams;

    .. member:: size_t PoisParams.m_states

       Integral value specifying the number of states in the HMM.

    .. member:: scalar *PoisParams.lambda

        Pointer to memory for state-dependend means. The object pointed to by
        :member:`PoisParams.lambda` must hold enough memory for
        :member:`PoisParams.m_states` values of type :type:`scalar`.

    .. member:: scalar *PoisParams.gamma

        Pointer to memory for transition probability matrix. The object pointed
        to by :member:`PoisParams.gamma` must hold enough memory for two times
        :member:`PoisParams.m_states` values of type :type:`scalar`.
    
    .. member:: scalar *PoisParams.delta

        Pointer to memory for initial distribution. The object pointed to by
        :member:`PoisParams.delta` must hold enough memory for
        :member:`PoisParams.m_states` values of type :type:`scalar`.


Object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: PoisParams *PoisParams_New (const size_t m_states)

   Allocate memory fro a new :struct:`PoisParams` structure for a HMM with
   :var:`m_states` states.


.. function:: PoisParams *PoisParams_NewFromFile (const char *fpath)

    Allocate a new :struct:`PoisParams` structure and initialize it with the
    data read form file. The data file in location path :var:`fpath` must
    conform to the data file specification.


.. function:: PoisParams *PoisParams_NewRandom (const size_t m_states)

    Allocate memory for a new :struct:`PoisParams` for a HMM with
    :var:`m_states` states and initialize it with random data. Internally calls
    :func:`PoisParams_SetLambdaRnd`, :func:`PoisParams_SetGammaRnd`, and
    :func:`PoisParams_SetDeltaRnd`.

.. macro:: PoisParams_Delete (this)

    Delete a :struct:`PoisParams` struct from memory. 


Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: extern void PoisParams_SetLambda (    \
    PoisParams *const restrict params,              \
    const scalar *const restrict lambda)            

    Copy data from :var:`lambda` to corresponding member of :var:`params`.


.. function:: extern void PoisParams_SetGamma ( \
    PoisParams *const restrict params,          \
    const scalar *const restrict gamma)

    Copy data from :var:`gamma` to corresponding member of :var:`params`.


.. function:: extern void PoisParams_SetDelta ( \
    PoisParams *const restrict params,          \
    const scalar *const restrict delta)

    Copy data from :var:`delta` to correspinding member of :var:`params`.


.. function:: void PoisParams_SetLambdaRnd (PoisParams *const restrict this)

    Sample the state-dependend means of :var:`this` uniformly from the
    interval [1, 100].


.. function:: void PoisParams_SetGammaRnd (PoisParams *const restrict this)

    Sample the transition probability matrix (tpm) of :var:`this` randomly.
    This function guaratees that each row of the tpm is indeed a discrete
    probability distribution.


.. function:: void PoisParams_SetDeltaRnd (PoisParams *const restrict this)

		Sample the initial distribution of :var:`this` randomly. This function
		guaratees that the initial distribution is indeed a discrete probability
		distributions. 


Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: extern void PoisParams_Copy ( \
    const PoisParams *const restrict this,  \
    PoisParams *const restrict other)

    Copy parameters from :var:`this` to :var:`other`.

.. function:: extern void PoisParams_CopyLog (  \
    const PoisParams *restrict this,            \
    PoisParams *restrict other)

    Copy paramters from :var:`this` to :var:`other` and transform :var:`gamma`
    and :var:`delta` to log domain.
    

