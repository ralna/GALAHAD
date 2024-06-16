WCP
===

.. module:: galahad.wcp

.. include:: wcp_intro.rst

.. include:: wcp_storage.rst

functions
---------

   .. function:: wcp.initialize()

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

          print_level : int
             the level of output required is specified by print_level.
          start_print : int
             any printing will start on this iteration.
          stop_print : int
             any printing will stop on this iteration.
          maxit : int
             at most maxit inner iterations are allowed.
          initial_point : int
             how to choose the initial point. Possible values are

             * **0**

               the values input in X, shifted to be at least prfeas
               from  their nearest bound, will be used

             * **1**

               the nearest point to the "bound average" 0.5(X_l+X_u)
               that  satisfies the linear constraints will be used.

          factor : int
             the factorization to be used. Possible values are

             * **0**

               automatic

             * **1**

               Schur-complement factorization

             * **2**

               augmented-system factorization.

          max_col : int
             the maximum number of nonzeros in a column of A which is
             permitted with the Schur-complement factorization.
          indmin : int
             an initial guess as to the integer workspace required by
             SBLS.
          valmin : int
             an initial guess as to the real workspace required by SBLS.
          itref_max : int
             the maximum number of iterative refinements allowed.
          infeas_max : int
             the number of iterations for which the overall
             infeasibility of the problem is not reduced by at least a
             factor ``required_infeas_reduction`` before the problem is
             flagged as infeasible (see required_infeas_reducti.
          perturbation_strategy : int
             the strategy used to reduce relaxed constraint bounds.
             Possible values are

             * **0**

               do not perturb the constraints

             * **1**

               reduce all perturbations by the same amount with
               linear reduction

             * **2**

               reduce each perturbation as much as possible with
               linear reduction

             * **3**

               reduce all perturbations by the same amount with
               superlinear  reduction

             * **4**

               reduce each perturbation as much as possible with
               superlinear  reduction.
          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are

             * **0**

               nothing restored

             * **1**

               scalar and vector parameters

             * **2**

               all parameters.

          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_p : float
             the required accuracy for the primal infeasibility.
          stop_d : float
             the required accuracy for the dual infeasibility.
          stop_c : float
             the required accuracy for the complementarity.
          prfeas : float
             initial primal variables will not be closer than prfeas
             from their bound.
          dufeas : float
             initial dual variables will not be closer than dufeas from
             their bounds.
          mu_target : float
             the target value of the barrier parameter. If mu_target is
             not positive, it will be reset to an appropriate value.
          mu_accept_fraction : float
             the complemtary slackness x_i.z_i will be judged to lie
             within an acceptable margin around its target value mu as
             soon as mu_accept_fraction * mu <= x_i.z_i <= ( 1 /
             mu_accept_fraction ) * mu; the perturbations will be
             reduced as soon as all of the complemtary slacknesses
             x_i.z_i lie within acceptable bounds. mu_accept_fraction
             will be reset to ensure that it lies in the interval (0,1].
          mu_increase_factor : float
             the target value of the barrier parameter will be
             increased by mu_increase_factor for infeasible constraints
             every time the perturbations are adjusted.
          required_infeas_reduction : float
             if the overall infeasibility of the problem is not reduced
             by at least a factor required_infeas_reduction over
             ``infeas_max`` iterations, the problem is flagged as
             infeasible (see infeas_max).
          implicit_tol : float
             any primal or dual variable that is less feasible than
             implicit_tol will be regarded as defining an implicit
             constraint.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SBLS for details (obsolete).
          pivot_tol_for_dependencies : float
             the threshold pivot used by the matrix factorization when
             attempting to detect linearly dependent constraints. See
             the documentation for SBLS for details (obsolete).
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints (obsolete).
          perturb_start : float
             the constraint bounds will initially be relaxed by
             ``perturb_start;`` this perturbation will subsequently be
             reduced to zero. If perturb_start < 0, the amount by which
             the bounds are relaxed will be computed automatically.
          alpha_scale : float
             the test for rank defficiency will be to factorize (
             alpha_scale I A^T ) ( A 0 ).
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer tha identical_bounds_tol will be reset to the
             average of their values.
          reduce_perturb_factor : float
             the constraint perturbation will be reduced as follows:
             * - if the variable lies outside a bound, the
             corresponding  perturbation will be reduced to
             reduce_perturb_factor * current pertubation + ( 1 -
             reduce_perturb_factor ) * violation
             * - otherwise, if the variable lies within
             insufficiently_feasible of its bound the pertubation will
             be reduced to reduce_perturb_multiplier * current
             pertubation
             * - otherwise if will be set to zero.
          reduce_perturb_multiplier : float
             see reduce_perturb_factor.
          insufficiently_feasible : float
             see reduce_perturb_factor.
          perturbation_small : float
             if the maximum constraint pertubation is smaller than
             perturbation_small and the violation is smaller than
             implicit_tol, the method will deduce that there is a
             feasible point but no interior.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          remove_dependencies : bool
             the equality constraints will be preprocessed to remove
             any linear dependencies if True.
          treat_zero_bounds_as_general : bool
             any problem bound with the value zero will be treated as
             if it were a general value if True.
          just_feasible : bool
             if ``just_feasible`` is True, the algorithm will stop as
             soon as a feasible point is found. Otherwise, the optimal
             solution to the problem will be found.
          balance_initial_complementarity : bool
             if ``balance_initial_complementarity`` is ``true.`` the
             initial complemetarity will be balanced.
          use_corrector : bool
             if ``use_corrector,`` a corrector step will be used.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          record_x_status : bool
             if ``record_x_status`` is True, the array inform.X_status
             will be allocated and the status of the bound constraints
             will be reported on exit.
          record_c_status : bool
             if ``record_c_status`` is True, the array inform.C_status
             will be allocated and the status of the general
             constraints will be reported on exit.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          fdc_options : dict
             default control options for FDC (see ``fdc.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).

   .. function:: wcp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
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
          dictionary of control options (see ``wcp.initialize``).

   .. function:: wcp.find_wcp(n, m, a_ne, A_val, c_l, c_u, x_l, x_u, x, y_l, y_u, z_l, z_u, g)

      Find a well-centered point for a given polyhedral set of linear 
      inequalities.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``wcp.load``.
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
      y_l : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y_l$
          associated with the lower general constraints, 
          $A x \geq c_l$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y_l=0$, suffices and will be adjusted accordingly.
      y_u : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y_u$
          associated with the upper general constraints, 
          $A x \leq c_u$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y_u=0$, suffices and will be adjusted accordingly.
      z_l : ndarray(n)
          holds the initial estimate of the dual variables $z_l$
          associated with the lower simple bound constraints, 
          $x \geq x_l$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z_l=0$, suffices and will be adjusted accordingly.
      z_u : ndarray(n)
          holds the initial estimate of the dual variables $z_u$
          associated with the upper simple bound constraints, 
          $x \leq x_u$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z_u=0$, suffices and will be adjusted accordingly.
      g : ndarray(n)
          holds an optional dual target vector, if this is required 
          (for experts); normally a vetor of zero suffices.

      **Returns:**

      x : ndarray(n)
          holds the values of the well-centred point $x$ after
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

   .. function:: [optional] wcp.information()

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
              its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
              has been violated.

            * **-4**

              The constraint bounds are inconsistent.

            * **-5**

              The constraints appear to have no feasible point.

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

               The step is too small to make further impact.

            * **-18**

              Too many iterations have been performed. This may happen if
              options['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              options['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          c_implicit : int
             the number of general constraints that lie on (one) of
             their bounds for feasible solutions.
          x_implicit : int
             the number of variables that lie on (one) of their bounds
             for all feasible solutions.
          y_implicit : int
             the number of Lagrange multipliers for general constraints
             that lie on (one) of their bounds for all feasible
             solutions.
          z_implicit : int
             the number of dual variables that lie on (one) of their
             bounds for all feasible solutions.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by WCP_solve.
          mu_final_target_max : float
             the largest target value on termination.
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linear dependent constraints.
          feasible : bool
             is the returned primal-dual "solution" strictly feasible?
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
                  to  factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
               clock_find_dependent : float
                  the clock time spent detecting linear dependencies.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to  factorization.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          fdc_inform : dict
             inform parameters for FDC (see ``fdc.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).

   .. function:: wcp.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/wcp/Python/test_wcp.py
   :code: python

This example code is available in $GALAHAD/src/wcp/Python/test_wcp.py .
