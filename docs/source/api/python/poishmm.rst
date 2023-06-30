.. default-domain: python

chainsaddiction.poishmm
-------------------------------------------------------------------------------

.. module:: chainsaddiction.poishmm

.. class:: chainsaddiction.PoisHmm

   .. attribute:: err
      :type: bool

      ``True``, if an error was encountered during parameter estimation, else ``False``.

   .. attribute:: m_states
      :type: int

      Number of states.

   .. attribute:: n_ter
      :type: int

      Number of iterations performed. 

   .. attribute:: aic 
      :type: float

      Akaike information criterion.
   
   .. attribute:: bic
      :type: float

      Bayesian information criterion.

   .. attribute:: llk
      :type: float

      Logarithm of the likelihood.

   .. attribute:: lambda_
      :type: numpy.ndarray

      Estimate of the state-dependend probabilities.

   .. attribute:: gamma_
      :type: numpy.ndarray

      Estimate of the transition probability matrix.

   .. attribute:: delta_
      :type: numpy.ndarray

      Estimate of the initial distribution.

   .. attribute:: lalpha
      :type: numpy.ndarray

      Logarithm of the forward probabilities for each observation (rows) and
      state (columns).

   .. attribute:: lbeta
      :type: numpy.ndarray

      Logarithm of the backwward probabilities for each observation (rows) and
      state (columns).

   .. attribute:: lcsp
      :type: numpy.ndarray

      Logarithm of the conditional state probabilities for each observation
      (rows) and states (columns).


.. function:: fit(n_obs, m_states, max_iter, sdm, tpm, distr, data)

   :arg int n_obs: Number of observations
   :arg int m_states: Number of states
   :arg int max_iter: Maximum number of iterations
   :arg np.ndarray sdm: State-depended means
   :arg np.ndarray tpm: Transition probability matrix
   :arg np.ndarray distr: Start/initial distribution
   :arg np.ndarray data: Input data set
   :return: Fitted HMM
   :rtype: PoisHmm

   Fit a HMM with Poisson-distributed states to :data:`data`.


.. function:: read_params(path)

   :arg str path: Path to file

   Read Poisson HMM parameters from file.
