GLRT
====

.. module:: galahad.glrt

.. include:: glrt_intro.rst

functions
---------

   .. function:: glrt.initialize()

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
          stopping_rule : int
             the stopping rule used (see below). Possible values are:

             * **1**

               stoping rule = norm of the step.

             * **2**

               stopping rule is norm of the step / $\sigma$.

             * **other** 

               stopping rule = 1.0.
 
          freq : int
             frequency for solving the reduced tri-diagonal problem.
          extra_vectors : int
             the number of extra work vectors of length n used.
          ritz_printout_device : int
             the unit number for writing debug Ritz values.
          stop_relative : float
             the iteration stops successfully when the gradient in the
             $M^{-1}$ norm is smaller than max( ``stop_relative`` * min( 1,
             ``stopping_rule`` ) * norm initial gradient, ``stop_absolute`` ).
          stop_absolute : float
             see stop_relative.
          fraction_opt : float
             an estimate of the solution that gives at least
             ``fraction_opt`` times the optimal objective value will be
             found.
          rminvr_zero : float
             the smallest value that the square of the M norm of the
             gradient of the objective may be before it is considered
             to be zero.
          f_0 : float
             the constant term, f0, in the objective function.
          unitm : bool
             is M the identity matrix ?.
          impose_descent : bool
             is descent required i.e., should $c^T x < 0$ ?.
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


   .. function:: glrt.load_options(options=None)

      Import control options into internal storage prior to solution.

      **Parameters:**

      options : dict, optional
          dictionary of control options (see ``glrt.initialize``).

   .. function:: glrt.solve_problem(status, n, power, weight, r, v)

      Find the global moinimizer of the regularized quadratic objective 
      function $r(x)$.

      **Parameters:**

      status : int
          holds the entry status. Possible values are
          
          * **1**

          an initial entry with r set to $g$.

          * **6**

          a restart entry with $p$ and $g$ unchanged but a larger weight 
          $\sigma$.

          * **other**

          the value returned from the previous call, see Returns below.
         
      n : int
          holds the number of variables.
      power : float
          holds the regularization power $p \geq 2$.
      weight : float
          holds the strinctly positive regularization weight $\sigma$.
      r : ndarray(n)
          holds the values of the linear term $g$ in the objective function
          when initial or return status = 1, 4 or 6 (see below).
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

          * **4**

          the iteration must be restarted by setting r to $g$,
          and the function recalled with status set to 4.

          * **<0**

          an error occurred, see ``status`` in ``glrt.information`` for
          further details.

      x : ndarray(n)
          holds the values of the approximate minimizer $x$.
      r : ndarray(n)
          holds the values of the gradient $g + Hx$ at the current $x$.
      v : ndarray(n)
          holds the return vector when return status = 2 or 3 (see above).

   .. function:: [optional] glrt.information()

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

              The restriction n > 0, $\sigma > 0$ or $p \geq 2$ has been 
              violated.

            * **-7**

              The objective function appears to be unbounded from below.
              This can only happen if $p = 2$, and in this case the 
              objective is unbounded along the arc x + t v, where x and v 
              are as returned by ``glrt.solve_problem``, as t goes to infinity.

            * **-15** 

              $M$ appears to be indefinite.

            * **-18**

              The iteration limit has been exceeded.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          iter_pass2 : int
             the total number of pass-2 iterations required.
          obj : float
             the value of the quadratic function.
          obj_regularized : float
             the value of the regularized quadratic function.
          multiplier : float
             the multiplier, $\sigma \|x\|^{p-2}$.
          xpo_norm : float
             the value of the norm $\|x\|_M$.
          leftmost : float
             an estimate of the leftmost generalized eigenvalue of the
             pencil $(H,M)$.
          negative_curvature : bool
             was negative curvature encountered ?.
          hard_case : bool
             did the hard case occur ?.

   .. function:: glrt.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/glrt/Python/test_glrt.py
   :code: python

This example code is available in $GALAHAD/src/glrt/Python/test_glrt.py .
