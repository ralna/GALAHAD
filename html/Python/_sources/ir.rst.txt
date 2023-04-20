IR
==

.. module:: galahad.ir


Given a sparse symmetric $n \times n$ matrix $A = a_{ij}$ and the 
factorization of $A$ found by the GALAHAD package SLS, the ``ir`` package 
**solves the system of linear equations $A x = b$ using
iterative refinement.**

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/ir.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: ir.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             controls level of diagnostic output.
          itref_max : int
             maximum number of iterative refinements allowed.
          acceptable_residual_relative : float
             refinement will cease as soon as the residual $\|Ax-b\|$
             falls below max( acceptable_residual_relative * $\|b\|$,
             acceptable_residual_absolute ).
          acceptable_residual_absolute : float
             see acceptable_residual_relative.
          required_residual_relative : float
             refinement will be judged to have failed if the residual
             $\|Ax-b\| \geq $ required_residual_relative * $\|b\|$. No
             checking if required_residual_relative < 0.
          record_residuals : bool
             record the initial and final residual.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] ir.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             reported return status:
             * 0 the solution has been found.
             * -1 an array allocation has failed.
             * -2 an array deallocation has failed.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          norm_initial_residual : float
             infinity norm of the initial residual.
          norm_final_residual : float
             infinity norm of the final residual.

   .. function:: ir.finalize()

     Deallocate all internal private storage.
