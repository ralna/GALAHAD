LSQP
====

.. module:: galahad.lsqp

.. include:: lsqp_intro.rst

.. include:: lsqp_storage.rst

functions
---------

   .. function:: lsqp.initialize()

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
             factor ``reduce_infeas`` before the problem is flagged as
             infeasible (see reduce_infeas).
          muzero_fixed : int
             the initial value of the barrier parameter will not be
             changed for the first muzero_fixed iterations.
          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are

             * **0**

               nothing restored

             * **1**

               scalar and vector parameters

             * **2**

               all parameters.

          indicator_type : int
             specifies the type of indicator function used. Possible
             values are

             * **1**

               primal indicator: constraint active if and only if the
               distance to nearest bound $\f$\leq\f$ ``indicator_p_tol``

             * **2**

               primal-dual indicator: constraint active if and only
               if the  distance to nearest bound $\f$\leq\f$
               ``indicator_tol_pd`` * **size** of  corresponding multiplier

             * **3**

               primal-dual indicator: constraint active if and only
               if the  distance to the nearest bound $\f$\leq\f$
               ``indicator_tol_tapia`` * distance to same bound at
               previous iteration.

          extrapolate : int
             should extrapolation be used to track the central path?
             Possible values

             * **0**

               never

             * **1**

               after the final major iteration

             * **2**

               at each major iteration (unused at present).

          path_history : int
             the maximum number of previous path points to use when
             fitting the data (unused at present).
          path_derivatives : int
             the maximum order of path derivative to use (unused at
             present).
          fit_order : int
             the order of (Puiseux) series to fit to the path data:
             $$\leq$0 to fit all data (unused at present).
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
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
             from their bounds.
          dufeas : float
             initial dual variables will not be closer than dufeas from
             their bounds.
          muzero : float
             the initial value of the barrier parameter. If muzero is
             not positive, it will be reset to an appropriate value.
          reduce_infeas : float
             if the overall infeasibility of the problem is not reduced
             by at least a factor reduce_infeas over ``infeas_max``
             iterations, the problem is flagged as infeasible (see
             infeas_max).
          potential_unbounded : float
             if W=0 and the potential function value is smaller than
             potential_unbounded * number of one-sided bounds, the
             analytic center will be flagged as unbounded.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SBLS for details.
          pivot_tol_for_dependencies : float
             the threshold pivot used by the matrix factorization when
             attempting to detect linearly dependent constraints. See
             the documentation for SBLS for details.
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints.
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer tha identical_bounds_tol will be reset to the
             average of their values.
          mu_min : float
             start terminal extrapolation when mu reaches mu_min.
          indicator_tol_p : float
             if ``indicator_type`` = 1, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound  $\leq$ ``indicator_p_tol``.
          indicator_tol_pd : float
             if ``indicator_type`` = 2, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound $\leq$ ``indicator_tol_pd`` * size of
             corresponding multiplier.
          indicator_tol_tapia : float
             if ``indicator_type`` = 3, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound $\leq$ ``indicator_tol_tapia`` * distance to
             same bound at previous iteration.
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
          getdua : bool
             if ``getdua,`` is True, advanced initial values are
             obtained for the dual variables.
          puiseux : bool
             If extrapolation is to be used, decide between Puiseux and
             Taylor series.
          feasol : bool
             if ``feasol`` is True, the final solution obtained will be
             perturbed so tha variables close to their bounds are moved
             onto these bounds.
          balance_initial_complentarity : bool
             if ``balance_initial_complentarity`` is True, the initial
             complemetarity is required to be balanced.
          use_corrector : bool
             if ``use_corrector,`` a corrector step will be used.
          array_syntax_worse_than_do_loop : bool
             if ``array_syntax_worse_than_do_loop`` is True, f77-style
             do loops will be used rather than f90-style array syntax
             for vector operations.
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
          sif_file_name : str
             name of generated SIF file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          fdc_options : dict
             default control options for FDC (see ``fdc.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).

   .. function:: lsqp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          dictionary of control options (see ``lsqp.initialize``).


   .. function:: lsqp.solve_qp(n, m, f, g, w, x0, A_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a solution to the quadratic program involving the
      separable quadratic objective function $s(x)$.

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
          ``lsqp.load``.
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

   .. function:: [optional] lsqp.information()

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

            * **-15**

              The Hessian $H$ appears not to be positive definite.

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
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          nbacts : int
             the total number of "wasted" function evaluations during
             the linesearch.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by LSQP_solve_qp.
          potential : float
             the value of the logarithmic potential function sum
             -log(distance to constraint boundary).
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linear dependent constraints.
          feasible : bool
             is the returned "solution" feasible?.
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
          fdc_inform : dict
             inform parameters for FDC (see ``fdc.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).


   .. function:: lsqp.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/lsqp/Python/test_lsqp.py
   :code: python

This example code is available in $GALAHAD/src/lsqp/Python/test_lsqp.py .
