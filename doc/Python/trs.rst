TRS
===

.. module:: galahad.trs

.. include:: trs_intro.rst

.. include:: trs_storage.rst

functions
---------

   .. function:: trs.initialize()

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

          dense_factorization : int
             should the problem be solved by dense factorization?
             Possible values are

             * **0** 

               sparse factorization will be used

             * **1** 

               dense factorization will be used
  
             * **other**

               the choice is made automatically depending on the
               dimension  & sparsity.
  
          new_h : int
             how much of $H$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1** 

               values but not indices have changed

             * **2** 

               values and indices have changed.

          new_m : int
             how much of $M$ has changed since the previous call.
             Possible values are

             * **0** 

               unchanged

             * **1** 

               values but not indices have changed

             * **2** 

               values and indices have changed.

          new_a : int
             how much of $A$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1**

               values but not indices have changed

             * **2**

               values and indices have changed.

          max_factorizations : int
             the maximum number of factorizations (=iterations)
             allowed. -ve implies no limit.
          inverse_itmax : int
             the number of inverse iterations performed in the "maybe
             hard" case.
          taylor_max_degree : int
             maximum degree of Taylor approximant allowed.
          initial_multiplier : float
             initial estimate of the Lagrange multipler.
          lower : float
             lower and upper bounds on the multiplier, if known.
          upper : float
             see lower.
          stop_normal : float
             stop when $| ||x|| - \Delta | \leq$ max( stop_normal *
             $\Delta$, stop_absolute_normal ).
          stop_absolute_normal : float
             see stop_normal.
          stop_hard : float
             stop when bracket on optimal multiplier <= stop_hard *
             max( bracket ends ).
          start_invit_tol : float
             start inverse iteration when bracket on optimal multiplier
             <= stop_start_invit_tol * max( bracket ends ).
          start_invitmax_tol : float
             start full inverse iteration when bracket on multiplier <=
             stop_start_invitmax_tol * max( bracket ends).
          equality_problem : bool
             is the solution is **required** to lie on the boundary
             (i.e., is the constraint an equality)?.
          use_initial_multiplier : bool
             ignore initial_multiplier?.
          initialize_approx_eigenvector : bool
             should a suitable initial eigenvector should be chosen or
             should a previous eigenvector may be used?.
          force_Newton : bool
             ignore the trust-region if $H$ is positive definite.
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
          definite_linear_solver : str
             definite linear equation solver.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          ir_options : dict
             default control options for IR (see ``ir.initialize``).

   .. function:: trs.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

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
          dictionary of control options (see ``trs.initialize``).

   .. function:: trs.load_m(n, M_type, M_ne, M_row, M_col, M_ptr, options=None)

      Import problem data for the scaling matrix $M$, if needed, 
      into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      M_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      M_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      M_row : ndarray(M_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      M_col : ndarray(M_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      M_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``trs.initialize``).

   .. function:: trs.load_a(m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data for the constraint matrix $A$, if needed, 
      into internal storage prior to solution.

      **Parameters:**

      m : int
          holds the number of constraints.
      A_type : string
          specifies the unsymmetric storage scheme used for the Hessian $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in 
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      A_row : ndarray(A_ne)
          holds the row indices of $A$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other schemes, 
          and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of $A$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when 
          the other storage schemes are used, and in this case can be None.
      A_ptr : ndarray(m+1)
          holds the starting position of $A$, as well as the total number 
          of entries, in the sparse row-wise storage scheme. 
          It need not be set when the other schemes are used, 
          and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``trs.initialize``).

   .. function:: trs.solve_problem(n, radius, f, g, H_ne, H_val, M_ne, M_val, m, A_ne, A_val)

      Find the global minimizer of the quadratic objective function $q(x)$
      within the intersection of the trust-region and affine constraints.

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
          ``trs.load``.
      M_ne : int
          holds the number of entries in the lower triangular part of 
          the scaling matrix $M$ if it is not the identity matrix. 
          Otherwise it should be None.
      M_val : ndarray(M_ne)
          holds the values of the nonzeros in the lower triangle of the scaling
          matrix $M$ in the same order as specified in the sparsity pattern in 
          ``trs.load_m`` if needed. Otherwise it should be None.
      m : int
          holds the number of constraints.
      A_ne : int
          holds the number of entries in the lower triangular part of 
          the constraint matrix $A$ if $m > 0$.
          Otherwise it should be None.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the lower triangle of the 
          constraint matrix $A$ in the same order as specified in the 
          sparsity pattern in ``trs.load_a`` if needed. 
          Otherwise it should be None.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          affine constraints, if any. 
          Absent if ``trs.load_a`` has not been called.

   .. function:: [optional] trs.information()

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
              The restriction n > 0, m > 0, radius > 0,
              or requirement that type contains
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

            * **-15** 

              $M$ does not appear to be strictly diagonally dominant

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-19**

              The CPU time limit has been reached. This may happen if
              options['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          factorizations : int
             the number of factorizations performed.
          max_entries_factors : long
             the maximum number of entries in the factors.
          len_history : int
             the number of $(||x||_M,\lambda)$ pairs in the history.
          obj : float
             the value of the quadratic function.
          x_norm : float
             the $M$-norm of $x$, $||x||_M$.
          multiplier : float
             the Lagrange multiplier corresponding to the trust-region
             constraint.
          pole : float
             a lower bound $\max(0,-\lambda_1)$, where $\lambda_1$
             is the left-most eigenvalue of $(H,M)$.
          dense_factorization : bool
             was a dense factorization used?.
          hard_case : bool
             has the hard case occurred?.
          time : dict
             dictionary containing timing information:
               total : float
                  total CPU time spent in the package.
               assemble : float
                  CPU time spent building $H + \lambda M$.
               analyse : float
                  CPU time spent reordering $H + \lambda M$ prior to
                  factorization.
               factorize : float
                  CPU time spent factorizing $H + \lambda M$.
               solve : float
                  CPU time spent solving linear systems inolving
                  $H + \lambda M$.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time spent building $H + \lambda M$.
               clock_analyse : float
                  clock time spent reordering $H + \lambda M$ prior to
                  factorization.
               clock_factorize : float
                  clock time spent factorizing $H + \lambda M$.
               clock_solve : float
                  clock time spent solving linear systems inolving
                  $H + \lambda M$.
          history : dict
             dictionary recording the history of the iterates:
               lambda : ndarray(100)
                  the values of $\lambda$ for the first min(100,
                  ``len_history``) iterations.
               x_norm : ndarray(100)
                  the corresponding values of $\|x(\lambda)\|_M$.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          ir_inform : dict
             inform parameters for IR (see ``ir.information``).

   .. function:: trs.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/trs/Python/test_trs.py
   :code: python

This example code is available in $GALAHAD/src/trs/Python/test_trs.py .
