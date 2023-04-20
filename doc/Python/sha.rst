SHA
===

.. module:: galahad.sha

The ``sha`` package **finds an approximation to a sparse Hessian**
using componentwise secant approximation.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/sha.pdf for a brief description of the
method employed and other details.

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
             the level of output required. <= 0 gives no output, = 1
             gives a one-line summary for every iteration, = 2 gives a
             summary of the inner iteration for each iteration, >= 3
             gives increasingly verbose (debugging) output.
          approximation_algorithm : int
             which approximation algorithm should be used?
             * 0 : unsymmetric (alg 2.1 in paper)
             * 1 : symmetric (alg 2.2 in paper)
             * 2 : composite (alg 2.3 in paper)
             * 3 : composite 2 (alg 2.2/3 in paper).
          dense_linear_solver : int
             which dense linear equation solver should be used?
             * 1 : Gaussian elimination
             * 2 : QR factorization
             * 3 : singular-value decomposition
             * 4 : singular-value decomposition with divide-and-conquer.
          max_sparse_degree : int
             the maximum sparse degree if the combined version is used.
          extra_differences : int
             if available use an addition extra_differences differences.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
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
             return status.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          max_degree : int
             the maximum degree in the adgacency graph.
          differences_needed : int
             the number of differences that will be needed.
          max_reduced_degree : int
             the maximum reduced degree in the adgacency graph.

   .. function:: sha.finalize()

     Deallocate all internal private storage.
