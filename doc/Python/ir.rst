IR
==

.. module:: galahad.ir

Given a sparse symmetric $n \times n$ matrix $A = a_{ij}$ and the 
factorization of $A$ found by the GALAHAD package SLS, the ``ir`` package 
**solves the system of linear equations $A x = b$ using
iterative refinement.**

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/ir.pdf for additional details.

method
------

Iterative refinement proceeds as follows. First obtain the floating-point
solution to $A x = b$ using the factors of $A$. Then iterate
until either the desired residual accuracy (or the iteration limit is
reached) as follows: evaluate the residual $r = b - A x$,
find the floating-point solution $\delta x$ to $A \delta x = r$,
and replace $x$ by $x + \delta x$.

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
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
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
             the return status.  Possible values are:

             * **0**

               The insertion or deletion was succesful.

             * **-1**

               An allocation error occurred. A message indicating the
               offending array is written on unit control['error'], and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-2**

               A deallocation error occurred.  A message indicating the
               offending array is written on unit control['error'] and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-11**

               Iterative refinement has not reduced the relative residual by 
               more than control['required_relative_residual'].
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          norm_initial_residual : float
             the infinity norm of the initial residual.
          norm_final_residual : float
             the infinity norm of the final residual.

   .. function:: ir.finalize()

     Deallocate all internal private storage.