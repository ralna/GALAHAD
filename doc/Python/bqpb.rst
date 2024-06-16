BQPB
====

.. module:: galahad.bqpb

.. include:: bqpb_intro.rst

.. include:: bqpb_storage.rst

functions
---------

   .. function:: bqpb.initialize()

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
             specifies the type of indicator function used. Possible values are

             * **1**

               primal indicator: a constraint is active if and only
               if  the distance to its nearest bound <= ``indicator_p_tol``.

             * **2**

               primal-dual indicator: a constraint is active if and only if the
               distance to its nearest bound <= ``indicator_tol_pd`` times the
               size of the corresponding multiplier.

             * **3**

               primal-dual indicator: a constraint is active if and
               only if  the distance to its nearest bound <=
               ``indicator_tol_tapia`` times the distance to same bound at the
               previous iteration.

          arc : int
             which residual trajectory should be used to aim from the
             current iterate to the solution. Possible values are

             * **1**

               the Zhang linear residual trajectory.

             * **2**

               the Zhao-Sun quadratic residual trajectory.

             * **3**

               the Zhang arc ultimately switching to the Zhao-Sun
               residual trajectory.
  
             * **4**

               the mixed linear-quadratic residual trajectory.

             * **5**

               the Zhang arc ultimately switching to the mixed
               linear-quadratic  residual trajectory.

          series_order : int
             the order of (Taylor/Puiseux) series to fit to the path
             data.
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          qplib_file_device : int
             specifies the unit number to write generated QPLIB file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_abs_p : float
             the required absolute and relative accuracies for the
             primal infeasibility.
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
          perturb_h : float
             ``perturb_h`` will be added to the Hessian.
          prfeas : float
             initial primal variables will not be closer than
             ``prfeas`` from their bounds.
          dufeas : float
             initial dual variables will not be closer than ``dufeas``
             from their bounds.
          muzero : float
             the initial value of the barrier parameter. If muzero is
             not positive, it will be reset to an appropriate value.
          tau : float
             the weight attached to primal-dual infeasibility compared
             to complementa when assessing step acceptance.
          gamma_c : float
             individual complementarities will not be allowed to be
             smaller than gamma_c times the average value.
          gamma_f : float
             the average complementarity will not be allowed to be
             smaller than gamma_f times the primal/dual infeasibility.
          reduce_infeas : float
             if the overall infeasibility of the problem is not reduced
             by at least a factor ``reduce_infeas`` over ``infeas_max``
             iterations, the problem is flagged as infeasible (see
             infeas_max).
          obj_unbounded : float
             if the objective function value is smaller than
             obj_unbounded, it will be flagged as unbounded from below.
          potential_unbounded : float
             if W=0 and the potential function value is smaller than
             ``potential_unbounded`` $\ast$ number of one-sided bounds,
             the analytic center will be flagged as unbounded.
          identical_bounds_tol : float
             any pair of constraint bounds $(c_l,c_u)$ or $(x_l,x_u)$
             that are closer than ``identical_bounds_tol`` will be
             reset to the average of their values.
          mu_pounce : float
             start terminal extrapolation when mu reaches mu_pounce.
          indicator_tol_p : float
             if ``indicator_type`` = 1, a constraint/bound will be
             deemed to be active if and only if the distance to its
             nearest bound <= ``indicator_p_tol``.
          indicator_tol_pd : float
             if ``indicator_type`` = 2, a constraint/bound will be
             deemed to be active if and only if the distance to its
             nearest bound <= ``indicator_tol_pd`` * size of
             corresponding multiplier.
          indicator_tol_tapia : float
             if ``indicator_type`` = 3, a constraint/bound will be
             deemed to be active if and only if the distance to its
             nearest bound <= ``indicator_tol_tapia`` * distance to
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
          treat_separable_as_general : bool
             if ``just_feasible`` is True, the algorithm will stop as
             soon as a feasible point is found. Otherwise, the optimal
             solution to the problem will be found.
          just_feasible : bool
             if ``treat_separable_as_general,`` is True, any
             separability in the problem structure will be ignored.
          getdua : bool
             if ``getdua,`` is True, advanced initial values are
             obtained for the dual variables.
          puiseux : bool
             decide between Puiseux and Taylor series approximations to
             the arc.
          every_order : bool
             try every order of series up to series_order?.
          feasol : bool
             if ``feasol`` is True, the final solution obtained will be
             perturbed so that variables close to their bounds are
             moved onto these bounds.
          balance_initial_complentarity : bool
             if ``balance_initial_complentarity`` is True, the initial
             complemetarity is required to be balanced.
          crossover : bool
             if ``crossover`` is True, cross over the solution to one
             defined by linearly-independent constraints if possible.
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
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          fit_options : dict
             default control options for FIT (see ``fit.initialize``).
          roots_options : dict
             default control options for ROOTS (see ``roots.initialize``).
          cro_options : dict
             default control options for CRO (see ``cro.initialize``).

   .. function:: bqpb.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

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
          dictionary of control options (see ``bqpb.initialize``).

   .. function:: bqpb.solve_qp(n, f, g, H_ne, H_val, x_l, x_u, x, z)

      Find a solution to the bound-constrained convex quadratic program 
      involving the quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
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
          ``bqpb.load``.
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
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: bqpb.solve_sldqp(n, f, g, w, x0, x_l, x_u, x, z)

      Find a solution to the bound-constrained convex quadratic program 
      involving the shifted least-distance objective function $s(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      w : ndarray(n)
          holds the values of the weights $w$ in the objective function.
      x0 : ndarray(n)
          holds the values of the shifts $x^0$ in the objective function.
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
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] bqpb.information()

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

            * **-18**

              Too many iterations have been performed. This may happen if
              options['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              options['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-20**

              The Hessian $H$ appears to be indefinite.

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
          threads : int
             the number of threads used.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by BQPB_solve.
          primal_infeasibility : float
             the value of the primal infeasibility.
          dual_infeasibility : float
             the value of the dual infeasibility.
          complementary_slackness : float
             the value of the complementary slackness.
          init_primal_infeasibility : float
             these values at the initial point (needed by GALAHAD_CCQP).
          init_dual_infeasibility : float
             see init_primal_infeasibility.
          init_complementary_slackness : float
             see init_primal_infeasibility.
          potential : float
             the value of the logarithmic potential function sum
             -log(distance to constraint boundary).
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linear dependent constraints.
          feasible : bool
             is the returned "solution" feasible?.
          checkpointsIter : int
             checkpointsIter(i) records the iteration at which the
             criticality measures first fall below $10^{-i-1}$, i = 0,
             ``..,`` 15 (-1 means not achieved).
          checkpointsTime : float
             checkpointsIter(i) records the time at which the
             criticality measures first fall below $10^{-i-1}$, i = 0,
             ``..,`` 15 (-1 means not achieved).
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
          fit_inform : dict
             return information from FIT (see ``fit.information``).
          roots_inform : dict
             return information from ROOTS (see ``roots.information``).
          cro_inform : dict
             inform parameters for CRO (see ``cro.information``).
          rpd_inform : dict
             inform parameters for RPD (see ``rpd.information``).

   .. function:: bqpb.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/bqpb/Python/test_bqpb.py
   :code: python

This example code is available in $GALAHAD/src/bqpb/Python/test_bqpb.py .
