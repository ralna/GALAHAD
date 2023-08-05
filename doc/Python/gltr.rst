GLTR
====

.. module:: galahad.gltr

.. include:: gltr_intro.rst

functions
---------

   .. function:: gltr.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          problem : int
             unit to write problem data into file problem_file.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output,

             * **1**

               gives a one-line summary for every iteration.

             * **2**

               gives a summary of the inner iteration for each iteration.

             * **>=3**

               gives increasingly verbose (debugging) output.

          itmax : int
             the maximum number of iterations allowed (-ve = no bound).
          Lanczos_itmax : int
             the maximum number of iterations allowed once the boundary
             has been encountered (-ve = no bound).
          extra_vectors : int
             the number of extra work vectors of length n used.
          ritz_printout_device : int
             the unit number for writing debug Ritz values.
          stop_relative : float
             the iteration stops successfully when the gradient in the
             $M^{-1}$ norm is smaller than max( ``stop_relative`` *
             norm initial gradient, ``stop_absolute`` ).
          stop_absolute : float
             see stop_relative.
          fraction_opt : float
             an estimate of the solution that gives at least
             ``fraction_opt`` times the optimal objective value will be
             found.
          f_min : float
             the iteration stops if the objective-function value is
             lower than f_min.
          rminvr_zero : float
             the smallest value that the square of the M norm of the
             gradient of the the objective may be before it is
             considered to be zero.
          f_0 : float
             the constant term, $f$, in the objective function.
          unitm : bool
             is $M$ the identity matrix ?.
          steihaug_toint : bool
             should the iteration stop when the Trust-region is first
             encountered ?.
          boundary : bool
             is the solution thought to lie on the constraint boundary ?.
          equality_problem : bool
             is the solution required to lie on the constraint boundary?.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          print_ritz_values : bool
             should the Ritz values be written to the debug stream?.
          ritz_file_name : str
             name of debug file containing the Ritz values.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.


   .. function:: gltr.load_options(options=None)

      Import control options into internal storage prior to solution.

      **Parameters:**

      options : dict, optional
          dictionary of control options (see ``gltr.initialize``).

   .. function:: gltr.solve_problem(status, n, radius, r, v)

      Find the global moinimizer of the quadratic objective function $q(x)$
      within a trust-region of radius $\Delta$.

      **Parameters:**

      status : int
          holds the entry status. Possible values are
          
          * **1**

          an initial entry with r set to $g$.

          * **4**

          a restart entry with $g$ unchanged but a smaller radius $\Delta$.

          * **other**

          the value returned from the previous call, see Returns below.
         
      n : int
          holds the number of variables.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      r : ndarray(n)
          holds the values of the linear term $g$ in the objective function
          when initial or return status = 1, 4 or 5  (see below).
      v : ndarray(n)
          holds the result vector when return status = 2 or 3 (see below).

      **Returns:**

      status : int
          holds the exit status. Possible values are
          
          * **0**

          the solution has been found, no further reentry is required

          * **2**

          the inverse of $M$ must be applied to the vector returned in v,
          the result placed in v, and the function recalled with status
          set to 2. This will only occur if options['unitm'] is False.

          * **3**

          the product of $H$ with the vector returned in v must be formed,
          the result placed in v, and the function recalled with status
          set to 3.

          * **5**

          the iteration must be restarted by setting r to $g$,
          and the function recalled with status set to 5.

          * **<0**

          an error occurred, see ``status`` in ``gltr.information`` for
          further details.

      x : ndarray(n)
          holds the values of the approximate minimizer $x$.
      r : ndarray(n)
          holds the values of the gradient $g + Hx$ at the current $x$.
      v : ndarray(n)
          holds the return vector when return status = 2 or 3 (see above).

   .. function:: [optional] gltr.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            return status.  Possible values are:

            * **0**

              The run was successful.

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

              The restriction n > 0 or $\Delta > 0$ has been violated.

            * **-15** 

              $M$ appears to be indefinite.

            * **-18**

              The iteration limit has been exceeded.

            * **-30**

              The trust-region has been encountered in Steihaug-Toint mode.

            * **-31** 

              The function value is smaller than options['f_min'].

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          iter_pass2 : int
             the total number of pass-2 iterations required if the
             solution lies on the trust-region boundary.
          obj : float
             the value of the quadratic function.
          multiplier : float
             the Lagrange multiplier corresponding to the trust-region
             constraint.
          mnormx : float
             the $M$-norm of $x$, $\|x\|_M$.
          piv : float
             the latest pivot in the Cholesky factorization of the
             Lanczos tridiagona.
          curv : float
             the most negative cuurvature encountered.
          rayleigh : float
             the current Rayleigh quotient.
          leftmost : float
             an estimate of the leftmost generalized eigenvalue of the
             pencil $(H,M)$.
          negative_curvature : bool
             was negative curvature encountered ?.
          hard_case : bool
             did the hard case occur ?.

   .. function:: gltr.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/gltr/Python/test_gltr.py
   :code: python

This example code is available in $GALAHAD/src/gltr/Python/test_gltr.py .
