QPA
===

.. module:: galahad.qpa

.. include:: qpa_intro.rst

.. include:: qpa_storage.rst

functions
---------

   .. function:: qpa.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
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
          maxit : int
             at most maxit inner iterations are allowed.
          factor : int
             the factorization to be used. Possible values are 0
             automatic 1 Schur-complement factorization 2
             augmented-system factorization.
          max_col : int
             the maximum number of nonzeros in a column of A which is
             permitted with the Schur-complement factorization.
          max_sc : int
             the maximum permitted size of the Schur complement before
             a refactorization is performed.
          indmin : int
             an initial guess as to the integer workspace required by
             SLS (OBSOLETE).
          valmin : int
             an initial guess as to the real workspace required by SLS
             (OBSOLETE).
          itref_max : int
             the maximum number of iterative refinements allowed
             (OBSOLETE).
          infeas_check_interval : int
             the infeasibility will be checked for improvement every
             infeas_check_interval iterations (see
             infeas_g_improved_by_factor and
             infeas_b_improved_by_factor below).
          cg_maxit : int
             the maximum number of CG iterations allowed. If cg_maxit <
             0, this number will be reset to the dimension of the
             system + 1.
          precon : int
             the preconditioner to be used for the CG is defined by
             precon. Possible values are 0 automatic 

             * **1**

               no preconditioner, i.e, the identity within full factorization 

             * **2**

               full factorization 

             * **3**

               band within full factorization 

             * **4**

               diagonal using the barrier terms within full factorization.

          nsemib : int
             the semi-bandwidth of a band preconditioner, if
             appropriate.
          full_max_fill : int
             if the ratio of the number of nonzeros in the factors of
             the reference matrix to the number of nonzeros in the
             matrix itself exceeds full_max_fill, and the
             preconditioner is being selected automatically (precon =
             0), a banded approximation will be used instead.
          deletion_strategy : int
             the constraint deletion strategy to be used. Possible
             values are:  0 most violated of all 1 LIFO (last in, first
             out) k LIFO(k) most violated of the last k in LIFO.
          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are 0 nothing
             restored 1 scalar and vector parameters 2 all parameters.
          monitor_residuals : int
             the frequency at which residuals will be monitored.
          cold_start : int
             indicates whether a cold or warm start should be made.
             Possible values are

             * **0**

               warm start - the values set in C_stat and B_stat indicate 
               which constraints will be included in the initial working set. 

             * **1**

               cold start from the value set in X; constraints active at X 
               will determine the initial working set. 

             * **2**

               cold start with no active constraints 

             * **3**

               cold start with only equality constraints active 

             * **4**

               cold start with as many active constraints as possible.

          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          feas_tol : float
             any constraint violated by less than feas_tol will be
             considered to be satisfied.
          obj_unbounded : float
             if the objective function value is smaller than
             obj_unbounded, it will b flagged as unbounded from below.
          increase_rho_g_factor : float
             if the problem is currently infeasible and solve_qp (see
             below) is True, the current penalty parameter for the
             general constraints will be increased by
             increase_rho_g_factor when needed.
          infeas_g_improved_by_factor : float
             if the infeasibility of the general constraints has not
             dropped by a fac of infeas_g_improved_by_factor over the
             previous infeas_check_interval iterations, the current
             corresponding penalty parameter will be increase.
          increase_rho_b_factor : float
             if the problem is currently infeasible and solve_qp or
             solve_within_boun (see below) is True,` the current
             penalty parameter for the simple boun constraints will be
             increased by increase_rho_b_factor when needed.
          infeas_b_improved_by_factor : float
             if the infeasibility of the simple bounds has not dropped
             by a factor of infeas_b_improved_by_factor over the
             previous infeas_check_interval iterations, the current
             corresponding penalty parameter will be increase.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SLS for details (OBSOLETE).
          pivot_tol_for_dependencies : float
             the threshold pivot used by the matrix factorization when
             attempting to detect linearly dependent constraints.
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints (OBSOLETE).
          inner_stop_relative : float
             the search direction is considered as an acceptable
             approximation to the minimizer of the model if the
             gradient of the model in the preconditioning(inverse) norm
             is less than max( inner_stop_relative * initial
             preconditioning(inverse) gradient norm,
             inner_stop_absolute ).
          inner_stop_absolute : float
             see inner_stop_relative.
          multiplier_tol : float
             any dual variable or Lagrange multiplier which is less
             than multiplier_t outside its optimal interval will be
             regarded as being acceptable when checking for optimality.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          treat_zero_bounds_as_general : bool
             any problem bound with the value zero will be treated as
             if it were a general value if True.
          solve_qp : bool
             if solve_qp is True, the value of prob.rho_g and
             prob.rho_b will be increased as many times as are needed
             to ensure that the output solution is feasible, and thus
             aims to solve the quadratic program (2)-(4).
          solve_within_bounds : bool
             if solve_within_bounds is True, the value of
             prob.rho_b will be increased as many times as are needed
             to ensure that the output solution is feasible with
             respect to the simple bounds, and thus aims to solve the
             bound-constrained quadratic program (4)-(5).
          randomize : bool
             if randomize is True, the constraint bounds will be
             perturbed by small random quantities during the first
             stage of the solution process. Any randomization will
             ultimately be removed. Randomization helps when solving
             degenerate problems.
          array_syntax_worse_than_do_loop : bool
             if ``array_syntax_worse_than_do_loop`` is True, f77-style
             do loops will be used rather than f90-style array syntax
             for vector operations (OBSOLETE).
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if ``generate_sif_file`` is True, a SIF file
             describing the current problem is to be generated.
          symmetric_linear_solver : str
             indefinite linear equation solver.
          definite_linear_solver : str
             definite linear equation solver.
          sif_file_name : str
             name of generated SIF file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          each_interval : bool
             component specifically for parametric problems (not used
             at present).
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).

   .. function:: qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
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
      A_type : string
          specifies the unsymmetric storage scheme used for the constraints 
          Jacobian $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in $A$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other two schemes.
      A_row : ndarray(A_ne)
          holds the row indices of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other two schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of $A$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when the 
          dense storage scheme is used, and in this case can be None.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries, in the sparse row-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      options : dict, optional
          dictionary of control options (see ``qpa.initialize``).

   .. function:: qpa.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the standard non-convex quadratic program 
      involving the quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
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
          ``qpa.load``.
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: qpa.solve_l1qp(n, m, f, g, H_ne, H_val, rho_g, rho_b, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the non-convex quadratic program involving the
      $\mathbf{\ell_1}$ quadratic objective function $f(x;\rho_g,\rho_b)$
      for given $\rho_g$. and $\rho_b$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
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
          ``qpa.load``.
      rho_g : float
          holds the weight $\rho_g$ associated with the general 
          infeasibilities
      rho_b : float
          holds the weight $\rho_b$ associated with the simple bound
          infeasibilities
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: qpa.solve_bcl1qp(n, m, f, g, H_ne, H_val, rho_g, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the non-convex quadratic program involving the
      bound-constrained $\mathbf{\ell_1}$ quadratic objective function 
      $q(x) + \rho_g v_g(x),$ for given $\rho_g$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
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
          ``qpa.load``.
      rho_g : float
          holds the weight $\rho_g$ associated with the general 
          infeasibilities
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] qpa.information()

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
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
              has been violated.

            * **-4**

              The bound constraints are inconsistent.

            * **-5**

              The constraints appear to have no feasible point.

            * **-7**

              The objective function appears to be unbounded from below
              on the feasible set.

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

            * **-17**

              The step is too small to make further progress.

            * **-18**

              Too many iterations have been performed. This may happen if
              options['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

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
          major_iter : int
             the total number of major iterations required.
          iter : int
             the total number of iterations required.
          cg_iter : int
             the total number of conjugate gradient iterations required.
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          nmods : int
             the total number of factorizations which were modified to
             ensure that th matrix was an appropriate preconditioner.
          num_g_infeas : int
             the number of infeasible general constraints.
          num_b_infeas : int
             the number of infeasible simple-bound constraints.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by QPA_solve.
          infeas_g : float
             the 1-norm of the infeasibility of the general constraints.
          infeas_b : float
             the 1-norm of the infeasibility of the simple-bound
             constraints.
          merit : float
             the merit function value = obj + rho_g * infeas_g + rho_b
             * infeas_b.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorizatio.
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
                  to factorizat.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).


   .. function:: qpa.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/qpa/Python/test_qpa.py
   :code: python

This example code is available in $GALAHAD/src/qpa/Python/test_qpa.py .
