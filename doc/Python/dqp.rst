DQP
===

.. module:: galahad.dqp

.. include:: dqp_intro.rst

.. include:: dqp_storage.rst

functions
---------

   .. function:: dqp.initialize()

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
          print_gap : int
             printing will only occur every print_gap iterations.
          dual_starting_point : int
             which starting point should be used for the dual problem.
             Possible values are

             * **-1**

               user supplied comparing primal vs dual variables.

             * **0**

               user supplied.

             * **1**

               minimize linearized dual.

             * **2**

               minimize simplified quadratic dual.

             * **3**

               all free (= all active primal costraints).

             * **4**

               all fixed on bounds (= no active primal costraints).

          maxit : int
             at most maxit inner iterations are allowed.
          max_sc : int
             the maximum permitted size of the Schur complement before
             a refactorization is performed (used in the case where
             there is no Fredholm Alternative, 0 = refactor every
             iteration).
          cauchy_only : int
             a subspace step will only be taken when the current Cauchy
             step has changed no more than than cauchy_only active
             constraints; the subspace step will always be taken if
             cauchy_only < 0.
          arc_search_maxit : int
             how many iterations are allowed per arc search (-ve = as
             many as require.
          cg_maxit : int
             how many CG iterations to perform per DQP iteration (-ve
             reverts to n+1).
          explore_optimal_subspace : int
             once a potentially optimal subspace has been found,
             investigate it

             * **0**

               as per an ordinary subspace.

             * **1**

               by increasing the maximum number of allowed CG
               iterations

             * **2**

               by switching to a direct method.

          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are

             * **0**

               nothing restored.

             * **1**

               scalar and vector parameters.

             * **2**

               all parameters.

          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          qplib_file_device : int
             specifies the unit number to write generated QPLIB file
             describing the current problem.
          rho : float
             the penalty weight, rho. The general constraints are not
             enforced explicitly, but instead included in the objective
             as a penalty term weighted by rho when rho > 0. If rho <=
             0, the general constraints are explicit (that is, there is
             no penalty term in the objective function).
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_abs_p : float
             the required absolute and relative accuracies for the
             primal infeasibilies.
          stop_rel_p : float
             see stop_abs_p.
          stop_abs_d : float
             the required absolute and relative accuracies for the dual
             infeasibility.
          stop_rel_d : float
             see stop_abs_d.
          stop_abs_c : float
             the required absolute and relative accuracies for the
             complementarity.
          stop_rel_c : float
             see stop_abs_c.
          stop_cg_relative : float
             the CG iteration will be stopped as soon as the current
             norm of the preconditioned gradient is smaller than max(
             stop_cg_relative * initial preconditioned gradient,
             stop_cg_absolute ).
          stop_cg_absolute : float
             see stop_cg_relative.
          cg_zero_curvature : float
             threshold below which curvature is regarded as zero if CG
             is used.
          max_growth : float
             maximum growth factor allowed without a refactorization.
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer than identical_bounds_tol will be reset to the
             average of their values.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          initial_perturbation : float
             the initial penalty weight (for DLP only).
          perturbation_reduction : float
             the penalty weight reduction factor (for DLP only).
          final_perturbation : float
             the final penalty weight (for DLP only).
          factor_optimal_matrix : bool
             are the factors of the optimal augmented matrix required?
             (for DLP only).
          remove_dependencies : bool
             the equality constraints will be preprocessed to remove
             any linear dependencies if True.
          treat_zero_bounds_as_general : bool
             any problem bound with the value zero will be treated as
             if it were a general value if True.
          exact_arc_search : bool
             if ``exact_arc_search`` is True, an exact piecewise arc
             search will be performed. Otherwise an ineaxt search using
             a backtracing Armijo strategy will be employed.
          subspace_direct : bool
             if ``subspace_direct`` is True, the subspace step will be
             calculated using a direct (factorization) method, while if
             it is False, an iterative (conjugate-gradient) method will
             be used.
          subspace_alternate : bool
             if ``subspace_alternate`` is True, the subspace step will
             alternate between a direct (factorization) method and an
             iterative (GLTR conjugate-gradient) method. This will
             override ``subspace_direct``.
          subspace_arc_search : bool
             if ``subspace_arc_search`` is True, a piecewise arc search
             will be performed along the subspace step. Otherwise the
             search will stop at the firstconstraint encountered.
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
          generate_qplib_file : bool
             if ``generate_qplib_file`` is True, a QPLIB file
             describing the current problem is to be generated.
          symmetric_linear_solver : str
             indefinite linear equation solver set in
             symmetric_linear_solver.
          definite_linear_solver : str
             definite linear equation solver.
          unsymmetric_linear_solver : str
             unsymmetric linear equation solver.
          sif_file_name : str
             name of generated SIF file containing input problem.
          qplib_file_name : str
             name of generated QPLIB file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          fdc_options : dict
             default control options for FDC (see ``fdc.initialize``).
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          gltr_options : dict
             default control options for GLTR (see ``gltr.initialize``).

   .. function:: dqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          dictionary of control options (see ``dqp.initialize``).

   .. function:: dqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a solution to the strictly convex quadratic program involving the
      quadratic objective function $q(x)$.

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
          ``dqp.load``.
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``dqp.load``.
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

   .. function:: dqp.solve_sldqp(n, m, f, g, w, x0, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a solution to the quadratic program involving the
      shifted least-distance objective function $s(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      w : ndarray(n)
          holds the values of the weights $w$ in the objective function.
      x0 : ndarray(n)
          holds the values of the shifts $x^0$ in the objective function.
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``cqp.load``.
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
      xstat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] dqp.information()

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

            * **-9**

              The analysis phase of the factorization failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              The factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

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

            * **-20**

              The Hessian $H$ appears not to be positive definite.

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          cg_iter : int
             the total number of iterations required.
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          threads : int
             the number of threads used.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by DQP_solve.
          primal_infeasibility : float
             the value of the primal infeasibility.
          dual_infeasibility : float
             the value of the dual infeasibility.
          complementary_slackness : float
             the value of the complementary slackness.
          non_negligible_pivot : float
             the smallest pivot that was not judged to be zero when
             detecting linearly dependent constraints.
          feasible : bool
             is the returned "solution" feasible?.
          checkpointsIter : ndarray(16)
             checkpointsIter(i) records the iteration at which the
             criticality measures first fall below 
             $10^{-i-1}, i = 0, \ldots 15$ (where -1 means not achieved).
          checkpointsTime : ndarray(16)
             checkpointsTime(i) records the CPU time at which the
             criticality measures first fall below 
             $10^{-i-1}, i = 0, \ldots 15$ (where -1 means not achieved).
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               find_dependent : float
                  the CPU time spent detecting linear dependencies.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               search : float
                  the CPU time spent in the linesearch.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
               clock_find_dependent : float
                  the clock time spent detecting linear dependencies.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorization.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
               clock_search : float
                  the clock time spent in the linesearch.
          fdc_inform : dict
             inform parameters for FDC (see ``fdc.information``).
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).
          gltr_inform : dict
             return information from GLTR (see ``gltr.information``).
          scu_status : int
             status value for SCU (see ``scu.status``).
          scu_inform : dict
             inform parameters for SCU (see ``scu.information``).
          rpd_inform : dict
             inform parameters for RPD (see ``rpd.information``).


   .. function:: dqp.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/dqp/Python/test_dqp.py
   :code: python

This example code is available in $GALAHAD/src/dqp/Python/test_dqp.py .
