.. default-domain: python

chainsaddiction.utils
-------------------------------------------------------------------------------

.. module:: chainsaddiction.utils


.. function:: local_decoding(lcsp) -> numpy.ndarray

   :arg numpy.ndarray lcsp: Log of conditional state probabilities
   :rtype: numpy.ndarray

   Compute the most likely state for each observation individually.


.. function:: global_decodung(lgamma, ldelta, lcsp) -> numpy.ndarray

   :arg numpy.ndarray lgamma: Log of transition probability matrix
   :arg numpy.ndarray ldelta: Log of initial distribution
   :arg numpy.ndarray lcsp: Log of conditional state probabilities
   :rtype: numpy.ndarray

   Compute the most likey *sequence* of hidden states given the sequence of
   observations and the estimated transition probailities.
