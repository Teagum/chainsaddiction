.. default-domain:: c

Probability Layer
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

        Integral, positiv number :member:`PoisProbs.n_obs`

    .. member:: size_t PoisProbs.m_states

    .. member:: scalar *PoisProbs.lsdp

    .. member:: scalar *PoisProbs.lalpha

    .. member:: scalar *PoisProbs.lbeta

    .. member:: scalar *PoisProbs.lcxpt


Object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
