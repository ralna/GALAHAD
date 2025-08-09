EXPO
====

.. module:: galahad.expo

.. include:: expo_intro.rst

.. include:: expo_storage.rst

functions
---------

   .. function:: expo.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required. Possible values are

             * **<= 0**

               gives no output.

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
          max_it : int
             the maximum number of iterations permitted.
          max_eval : int
             the maximum number of function evaluations permitted.
          alive_unit : int
             removal of the file alive_file from unit alive_unit
             terminates execution.
          alive_file : str
             see alive_unit.
          update_multipliers_itmin : int
             update the Lagrange multipliers/dual variables from iteration
             update_multipliers_itmin (<0 means never) and once the primal
             infeasibility is below update_multipliers_tol
          update_multipliers_tol : float
             see update_multipliers_itmin.
          infinity : float
             any bound larger than infinity in modulus will be regarded as
             infinite.
          stop_abs_p : float
             the required absolute accuracy for the primal infeasibility.
          stop_rel_p : float
             the required relative accuracy for the primal infeasibility.
          stop_abs_d : float
             the required absolute accuracy for the dual infeasibility.
          stop_rel_d : float
             the required relative accuracy for the dual infeasibility.
          stop_abs_c : float
             the required absolute accuracy for the complementarity.
          stop_rel_c : float
             the required relative accuracy for the complementarity.
          stop_s : float
             the smallest the norm of the step can be before termination.
          stop_subproblem_rel : float
              the subproblem minimization that uses GALAHAD TRU will be 
              stopped as  soon as the relative decrease in the subproblem 
              gradient falls below .stop_subproblem_rel. If 
              .stop_subproblem_rel is 1.0 or bigger or 0.0 or smaller, 
              this value will be ignored, and the choice of stopping 
              rule delegated to .control_tru.stop_g_relative (see below)
          initial_mu : float
             initial value for the penalty parameter (<=0 means set
             automatically)
          mu_reduce : float
             the amount by which the penalty parameter is decreased
          obj_unbounded : float
            the smallest value the objective function may take before the
            problem is marked as unbounded.
          try_advanced_start : float
            try an advanced start at the end of every iteration when the KKT
            residuals are smaller than .try_advanced_start (-ve means never)
          try_sqp_start : float
            try an advanced SQP start at the end of every iteration when the
            KKT residuals are smaller than .try_sqp_start (-ve means never)
          stop_advanced_start : float
            stop the advanced start search once the residuals small than
            .stop_advanced_start
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          hessian_available : bool
            is the Hessian matrix of second derivatives available or is
            access only via matrix-vector products (coming soon)?.
          subproblem_direct : bool
            use a direct (factorization) or (preconditioned) iterative
            method (coming soon) to find the search direction.
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
          bsc_options : dict
             default control options for BSC (see ``bsc.initialize``).
          tru_options : dict
             default control options for TRU (see ``tru.initialize``).
          ssls_options : dict
             default control options for SSLS (see ``ssls.initialize``).

   .. function:: expo.load(n, m, J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne,                          H_row, H_col, H_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      J_type : string
          specifies the unsymmetric storage scheme used for the Jacobian
          $J = J(x)$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      J_ne : int
          holds the number of entries in $J$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other two schemes.
      J_row : ndarray(J_ne)
          holds the row indices of $J$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other two schemes, and in this case can be None.
      J_col : ndarray(J_ne)
          holds the column indices of $J$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when the 
          dense storage scheme is used, and in this case can be None.
      J_ptr : ndarray(m+1)
          holds the starting position of each row of $J$, as well as the 
          total number of entries, in the sparse row-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      H_type : string, optional
          specifies the symmetric storage scheme used for the Hessian 
          $Hl = H(x,y)$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense' or
          'diagonal'; lower or upper case variants are allowed.
          This and the following H_* arguments are only required if
          a Newton approximation or tensor Gauss-Newton approximation
          model is required (see control.model = 4,...,8).
      H_ne : int, optional
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other three schemes.
      H_row : ndarray(H_ne), optional
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other three schemes, and in this case can be None.
      H_col : ndarray(H_ne), optional
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the dense or diagonal
          storage schemes are used, and in this case can be None.
      H_ptr : ndarray(n+1), optional
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``expo.initialize``).

   .. function:: expo.solve(n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, eval_fc, eval_gj, eval_hl)

      Find an approximate  minimizer of a given constrained optimization
      problem using an exponential penalty method.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
      J_ne : int
          holds the number of entries in the Jacobian $J = J(x)$.
      H_ne : int, optional
          holds the number of entries in the lower triangular part of 
          the Hessian $H = H(x,y)$.
      c_l : ndarray(m)
          holds the values $c^l$ of the lower bounds on the
          constraints $c(x)$.
      c_u : ndarray(m)
          holds the values $c^u$ of the upper bounds on the
          constraints $c(x)$.
      x_l : ndarray(n)
          holds the values $x^l$ of the lower bounds on the
          optimization variables $x$.
      x_u : ndarray(n)
          holds the values $x^u$ of the upper bounds on the
          optimization variables $x$.
      x : ndarray(n)
          holds the initial values of optimization variables $x$.
      eval_fc : callable
          a user-defined function that must have the signature:

          ``f, c = eval_fc(x)``

          The value of the objective $f(x)$ and components of the constraints
          $c(x)$ evaluated at $x$ must be assigned to ``f`` and ``c``,
          respectively.
      eval_gj : callable
          a user-defined function that must have the signature:

          ``g, j = eval_gj(x)``

          The components of the gradient $g(x)$ and the nonzeros 
          in the Jacobian $J(x)$ of the constraint functions evaluated at
          $x$ must be assigned to ``g``  and ``j``, respectively,
          the latter in the same order as specified in the sparsity pattern 
          in ``expo.load``.
      eval_hl : callable
          a user-defined function that must have the signature:

          ``h = eval_hl(x,y)``

          The components of the nonzeros in the lower triangle of the Hessian
          $Hl(x,y)$ evaluated at $x$ and $y$ must be assigned to ``h`` in the 
          same order as specified in the sparsity pattern in ``expo.load``.

      **Returns:**

      x : ndarray(n)
          holds the value of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the value of the Lagrange multipliers $y$ after 
          a successful call.
      z : ndarray(n)
          holds the value of the dual variables $z$ after a successful call.
      c : ndarray(m)
          holds the value of the constraints $c(x)$.
      gl : ndarray(n)
          holds the gradient $gl(x,y,z)$ of the Lagrangian function.


   .. function:: [optional] expo.information()

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

              The restriction n > 0 or m > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal' or 'absent' has been violated.

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

              The preconditioner $S(x)$ appears not to be positive definite.

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-18**

              Too many iterations have been performed. This may happen if
              options['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              options['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-82**

              The user has forced termination of the solver by removing
              the file named options['alive_file'] from unit
              options['alive_unit'].

             
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          bad_eval : str
             the name of the user-supplied evaluation routine for which
             an error occurred.
          iter : int
             the total number of iterations performed.
          fc_eval : int
             the total number of evaluations of the objective function $f(x)$
             and constraint functions $c(x)$.
          gj_eval : int
             the total number of evaluations of the gradient $g(x)$ of $f(x)$
             and Jacobian $J(x)$ of $c(x)$.
          hl_eval : int
             the total number of evaluations of the Hessian $Hl(x,y)$ of
             the Lagrangian.
          obj : float
             the value of the objective function $f(x)$ at the best estimate
             of the solution, x, determined by EXPO_solve.
          primal_infeasibility : float
             the norm of the primal infeasibility at the best estimate of
             the solution x, determined by EXPO_solve.
          dual_infeasibility : float
             the norm of the dual infeasibility at the best estimate of
             the solution x, determined by EXPO_solve.
          complementary_slackness : float
             the norm of the complementary_slackness at the best estimate of
             the solution x, determined by EXPO_solve.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorization.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          bsc_inform : dict
             inform parameters for BSC (see ``bsc.information``).
          tru_inform : dict
             inform parameters for TRU (see ``tru.information``).
          ssls_inform : dict
             inform parameters for SSLS (see ``ssls.information``).

   .. function:: expo.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/expo/Python/test_expo.py
   :code: python

This example code is available in $GALAHAD/src/expo/Python/test_expo.py .
