 .. default-domain:: c

Examples
===============================================================================


Parameter estimation with Poisson HMMs
-------------------------------------------------------------------------------

The following code is a complete example program that estimates the parameters
of a three-states HMM given a hypothetical data set stored in `dataset.txt`
and a set of initial parameters stored in `init_params.txt`.


.. code-block:: c

   #include <stdlib.h>
   #include "chainsaddiction.h"

   int
   main (void)
   {
      const char data_path[]   = "path/to/dataset.txt";
      const char params_path[] = "path/to/init_params.txt";
            int  status        = 0;

      DataSet    *data  = NULL;
      PoisParams *initp = NULL;
      PoisHmm    *hmm   = NULL;

      data  = ds_NewFromFile (data_path);
      initp = PoisParams_NewFromFile (params_path);
      if (data == NULL || initp == NULL) 
      {
         status = 1;
         goto cleanup;
      }

      hmm = PoisHmm_New (data->size, initp->m_states);
      if (hmm == NULL)
      {
         status = 1;
         goto cleanup;
      }

      PoisHmm_Init (hmm, initp->lambda, initp->gamma, initp->delta);
      PoisHmm_EstimateParams (hmm, data);
      PoisHmm_Summary (hmm);

   cleanup:
      ds_FREE (data);
      PoisParams_Delete (initp);
      PoisHmm_Delete (hmm);
       
      return status ? EXIT_FAILURE : EXIT_SUCCESS; 
   }
