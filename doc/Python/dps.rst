DPS
===

.. module:: galahad.dps

.. include:: dps_intro.rst

.. include:: dps_storage.rst

functions
---------

   .. function:: dps.initialize()

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

          new_h : int
             how much of $H$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged.

             * **1**

               values but not indices have changed.

             * **2**

               values and indices have changed.

          taylor_max_degree : int
             maximum degree of Taylor approximant allowed.
          eigen_min : float
             smallest allowable value of an eigenvalue of the block
             diagonal factor of $H$.
          lower : float
             lower and upper bounds on the multiplier, if known.
          upper : float
             see lower.
          stop_normal : float
             stop trust-region solution when
             $| \|x\|_M - \Delta | \leq$ max( ``stop_normal`` * delta,
             ``stop_absolute_normal`` ).
          stop_absolute_normal : float
             see stop_normal.
          goldfarb : bool
             use the Goldfarb variant of the
             trust-region/regularization norm rather than the modified
             absolute-value version.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          problem_file : str
             name of file into which to write problem data.
          symmetric_linear_solver : str
             symmetric (indefinite) linear equation solver.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).

   .. function:: dps.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      H_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      H_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      H_row : ndarray(H_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      H_col : ndarray(H_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``dps.initialize``).

   .. function:: dps.solve_tr_problem(n, radius, f, g, H_ne, H_val)

      Find the global moinimizer of the quadratic objective function $q(x)$
      within the trust-region.

      **Parameters:**

      n : int
          holds the number of variables.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      H_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(H_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``dps.load``.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.

   .. function:: rqs.solve_rq_problem(n, weight, power, f, g, H_ne, H_val)

      Find the global moinimizer of the regularized quadratic objective 
      function $r(x)$

      **Parameters:**

      n : int
          holds the number of variables.
      weight : float
          holds the strictly positive regularization weight, $\sigma$.
      power : float
          holds the regularization power, $p \geq 2$.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      H_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(H_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``rqs.load``.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.

   .. function:: dps.resolve_tr_problem(n, radius, f, g)

      Find the global moinimizer of the quadratic objective function $q(x)$
      within the trust-region  after any of the non-matrix terms has changed.

      **Parameters:**

      n : int
          holds the number of variables.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.

   .. function:: rqs.resolve_rq_problem(n, weight, power, f, g)

      Find the global moinimizer of the regularized quadratic objective 
      function $r(x)$ after any of the non-matrix terms has changed.

      **Parameters:**

      n : int
          holds the number of variables.
      weight : float
          holds the strictly positive regularization weight, $\sigma$.
      power : float
          holds the regularization power, $p \geq 2$.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.

   .. function:: [optional] dps.information()

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

             The restriction n > 0, radius > 0, weight > 0 or 
             power $\geq$ 2, or requirement that type contains
             its relevant string 'dense', 'coordinate', 'sparse_by_rows',
             'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
             has been violated.

           * **-9**

             The analysis phase of the factorization failed; the return
             status from the factorization package is given by
             inform['factor_status'].

           * **-10**

             The factorization failed; the return status from the
             factorization package is given by inform['factor_status'].

           * **-11**

             The solution of a set of linear equations using factors
             from the factorization package failed; the return status
             from the factorization package is given by
             inform['factor_status'].

           * **-16**

             The problem is so ill-conditioned that further progress
             is impossible.

           * **-23** 

             An entry from the strict upper triangle of $H$ has been 
             specified.

           * **-40** 

             An error occured when building $M$.

         alloc_status : int
            the status of the last attempted allocation/deallocation.
         bad_alloc : str
            the name of the array for which an allocation/deallocation
            error occurred.
         mod_1by1 : int
            the number of 1 by 1 blocks from the factorization of $H$
            that were modified when constructing $M$.
         mod_2by2 : int
            the number of 2 by 2 blocks from the factorization of $H$
            that were modified when constructing $M$.
         obj : float
            the value of the quadratic function.
         obj_regularized : float
            the value of the regularized quadratic function.
         x_norm : float
            the $M$-norm of the solution.
         multiplier : float
            the Lagrange multiplier associated with the
            constraint/regularization.
         pole : float
            a lower bound max(0,-lambda_1), where lambda_1 is the
            left-most eigenvalue of $(H,M)$.
         hard_case : bool
            has the hard case occurred?.
         time : dict
            dictionary containing timing information:
              total : float
                 total CPU time spent in the package.
              analyse : float
                 CPU time spent reordering H prior to factorization.
              factorize : float
                 CPU time spent factorizing H.
              solve : float
                 CPU time spent solving the diagonal model system.
              clock_total : float
                 total clock time spent in the package.
              clock_analyse : float
                 clock time spent reordering H prior to factorization.
              clock_factorize : float
                 clock time spent factorizing H.
              clock_solve : float
                 clock time spent solving the diagonal model system.
         sls_inform : dict
            inform parameters for SLS (see ``sbls.information``).

   .. function:: dps.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/dps/Python/test_dps.py
   :code: python

This example code is available in $GALAHAD/src/dps/Python/test_dps.py .
