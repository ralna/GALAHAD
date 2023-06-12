SHA
===

.. module:: galahad.sha

.. include:: sha_intro.rst

functions
---------

   .. function:: sha.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required. Possible values are:

             * **<=0** 

               no output.

             * **1** 

               a one-line summary for every iteration.

             * **2** 

               a summary of the inner iteration for each iteration.

             * **>=3** 

               increasingly verbose (debugging) output.
          approximation_algorithm : int
             which approximation algorithm should be used? Possible values are:
              
             * **0**
              
               unsymmetric (alg 2.1 in paper).
              
             * **1**
              
               symmetric (alg 2.2 in paper).

             * **2**

               composite (alg 2.3 in paper).

             * **3**

               composite 2 (alg 2.2/3 in paper).

          dense_linear_solver : int
             which dense linear equation solver should be used?
             Possible values are:

             * **1**

               Gaussian elimination.

             * **2**

               QR factorization.

             * **3**

               singular-value decomposition.

             * **4**

               singular-value decomposition with divide-and-conquer.
          max_sparse_degree : int
             the maximum sparse degree if the combined version is used.
          extra_differences : int
             if available use an addition extra_differences differences.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to use
             as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] sha.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            the return status.  Possible values are:

            * **0**

              The call was succesful.

            * **-1**

              An allocation error occurred. A message indicating the
              offending array is written on unit options['error'], and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-2**

              A deallocation error occurred.  A message indicating the
              offending array is written on unit options['error'] and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-3**

              The restriction n > 0 or nz >= 0 has been violated.

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package options['dense_linear_solver']
              failed when forming row ``bad_row``.

            * **-31**

              The call to ``sha_estimate`` was not preceded by a call to 
              ``sha_analyse``.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          max_degree : int
             the maximum degree in the adgacency graph.
          differences_needed : int
             the number of differences that will be needed.
          max_reduced_degree : int
             the maximum reduced degree in the adgacency graph.
          bad_row : int
             a failure occured when forming the bad_row-th row
             (0 = no failure).

   .. function:: sha.finalize()

     Deallocate all internal private storage.
