LSTR
====

.. module:: galahad.lstr

.. include:: lstr_intro.rst

functions
---------

   .. function:: lstr.initialize()

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

          start_print : int
             any printing will start on this iteration.
          stop_print : int
             any printing will stop on this iteration.
          print_gap : int
             the number of iterations between printing.
          itmin : int
             the minimum number of iterations allowed (-ve = no bound).
          itmax : int
             the maximum number of iterations allowed (-ve = no bound).
          itmax_on_boundary : int
             the maximum number of iterations allowed once the boundary
             has been encountered (-ve = no bound).
          bitmax : int
             the maximum number of Newton inner iterations per outer
             iteration allowe (-ve = no bound).
          extra_vectors : int
             the number of extra work vectors of length n used.
          stop_relative : float
             the iteration stops successfully when $\|A^Tr\|$ is less
             than max( ``stop_relative`` * $\|A^T b \|$,
             ``stop_absolute`` ), where $r = A x - b$.
          stop_absolute : float
             see stop_relative.
          fraction_opt : float
             an estimate of the solution that gives at least
             ``fraction_opt`` times the optimal objective value will be
             found.
          time_limit : float
             the maximum elapsed time allowed (-ve means infinite).
          steihaug_toint : bool
             should the iteration stop when the Trust-region is first
             encountered?.
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


   .. function:: lstr.load_options(options=None)

      Import control options into internal storage prior to solution.

      **Parameters:**

      options : dict, optional
          dictionary of control options (see ``lstr.initialize``).

   .. function:: lstr.solve_problem(status, m, n, radius, u, v)

      Find the global moinimizer of the quadratic objective function $q(x)$
      within a trust-region of radius $\Delta$.

      **Parameters:**

      status : int
          holds the entry status. Possible values are
          
          * **1**

          an initial entry with u set to $b$.

          * **5**

          a restart entry with u reset to $b$, but a smaller radius $\Delta$.

          * **other**

          the value returned from the previous call, see Returns below.
         
      m : int
          holds the number of residuals, i.e., the number of rows of $A$.
      n : int
          holds the number of variables, i.e., the number of columns of $A$.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      u : ndarray(m)
          holds the result vector when initial or return status = 1, 2, 
          4 or 5 (see below).
      v : ndarray(n)
          holds the result vector when return status = 3 (see below).

      **Returns:**

      status : int
          holds the exit status. Possible values are
          
          * **0**

          the solution has been found, no further reentry is required

          * **2**

          the sum $u + A v$, involving the vectors $u$ and $v$ returned in 
          u and v,  must be formed, the result placed in u, and the function 
          recalled with status set to 2.

          * **3**

          the sum $v + A^T u$, involving the vectors $u$ and $v$ returned in 
          u and v,  must be formed, the result placed in v, and the function 
          recalled with status set to 3.

          * **4**

          the iteration must be restarted by setting u to $b$,
          and the function recalled with status set to 4.

          * **<0**

          an error occurred, see ``status`` in ``lstr.information`` for
          further details.

      x : ndarray(n)
          holds the values of the approximate minimizer $x$.
      u : ndarray(m)
          holds the result vector $u$.
      v : ndarray(n)
          holds the result vector $v$.

   .. function:: [optional] lstr.information()

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

              The restriction n > 0, m > 0 or $\Delta > 0$ has been violated.

            * **-18**

              The iteration limit has been exceeded.

            * **-25**

              status is negative on entry.

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
          biters : int
             the total number of inner iterations performed.
          biter_min : int
             the smallest number of inner iterations performed during
             an outer iteration.
          biter_max : int
             the largestt number of inner iterations performed during
             an outer iteration.
          multiplier : float
             the Lagrange multiplier, $\lambda$, corresponding to the
             trust-region constraint.
          x_norm : float
             the Euclidean norm of $x$.
          r_norm : float
             the Euclidean norm of $Ax-b$.
          Atr_norm : float
             the Euclidean norm of $A^T (Ax-b) + \lambda x$.
          biter_mean : float
             the average number of inner iterations performed during an
             outer iteration.

   .. function:: lstr.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/lstr/Python/test_lstr.py
   :code: python

This example code is available in $GALAHAD/src/lstr/Python/test_lstr.py .
