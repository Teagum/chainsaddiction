 .. default-domain:: c

PoisHmm object
-------------------------------------------------------------------------------

.. struct:: PoisHmm

    The :struct:`PoisHmm` structure encapsulates all data of a
    Hidden-Markov model with Poisson-distributed states.

    .. code-block:: c
        
        typedef struct PoisHmm {
            bool err;
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

    .. member:: bool err

        A boolean error indicator. `err` equals `true` if an error occurred
        during fitting.

    .. member:: size_t n_obs

        An integral number greater than zero that specifies the number of
        states in an HMM.

    .. member:: size_t m_states

        An integral number that specifies the number of observations, that is,
        the length of the input data set.

    .. member:: size_t n_iter

        An integral number specifying the number of the iterations it took the
        fitting algorithm to converge. If it did not converge, this is equal to
        :var:`max_iter`.

    .. member:: size_t max_iter

        An positive integral number specifying the maximum number of iterations of
        the EM algorithm.

    .. member:: scalar tol

        An floating point number specifying the update tolerance of the fitting
        algorithm. An computed update is only applied if the update score is
        greater than :var:`tol`. 

    .. member:: scalar aic

        Storage for the Akaike Information Criterion of the fitted model.

    .. member:: scalar bic

        Storage for the Bayseian Information Criterion of the fitted model.

    .. member:: scalar llh

        Storage for the log likelihood of the fitted model.

    .. member:: PoisParams *init

        Pointer to initial model parameters.

    .. member:: PoisParams *params

        Pointer to estimated model parameters.

    .. member:: PoisProbs *probs

        Pointer to computation buffers.


Object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: PoisHmm *PoisHmm_New (    \
    const size_t n_obs,                 \
    const size_t m_states)

    Allocate memory for a new :struct:`PoisHmm` structure.

.. macro:: PoisHmm_Delete(this)

    Delete a :struct:`PoisHmm` struct from memory.


Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void PoisHmm_Init (           \
    PoisHmm *const restrict this,           \
    const scalar *const restrict lambda,    \
    const scalar *const restrict gamma,     \
    const scalar *const restrict delta)
    
    Initialize the parameters of a :struct:`PoisHmm` structure with starting values.

.. function:: void PoisHmm_InitRandom (PoisHmm *const restrict this)

    Initialize the parameters of a :struct:`PoisHmm` with a random parameters.


Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All operations require a properly initialized :struct:`PoisHmm` structure as
first parameter.

.. function:: void PoisHmm_EstimateParams (     \
    PoisHmm *const restrict this,               \
    const DataSet *const restrict inp)

    Compute maximum-likelihood estimates for the HMM parameters given the data
    set pointed to by :var:`inp`. Estimates are computed using the `Baum-Welch
    algorithm`_.

    During the fit, keep the members :member:`n_iter`, :member:`llh`,
    :member:`params`, and :member:`probs` up to date. This information may be
    used for further processing, such as model checking or, in case of errors,
    debugging at any time.

    Also, set the error indicator to `true` if the fit or any intermediate
    computation fails.

.. function:: int PoisHmm_ForwardBackward (PoisHmm *const restrict this)

   Compute the forward and backward probabilities of the HMM using the
   `forward-backward algorithm`_.

.. function:: int PoisHmm_ForwardProbabilities (PoisHmm *const restrict this)

    Compute only the forward probabilities under the HMM. 

.. function:: int PoisHmm_BackwardProbabilities (PoisHmm *const restrict this)

    Compute only the backward probabilities under the HMM.
    
.. function:: void PoisHmm_LogLikelihood (PoisHmm *const restrict this)

    Compute the logarithm of the data likelihood under the HMM.

.. function:: void PoisHmm_LogCondStateProbs (PoisHmm *const restrict this)

    Compute the logarithm of the conditional state probabilities.


Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void PoisHmm_Summary (const PoisHmm *const restrict this)

    Print estimated parameters and quality measures to stderr.


.. _Baum-Welch algorithm: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

.. _forward-backward algorithm: https://en.wikipedia.org/wiki/Forward-backward_algorithm
