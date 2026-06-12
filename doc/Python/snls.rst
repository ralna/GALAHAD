SNLS
====

.. module:: galahad.snls

.. include:: snls_intro.rst

.. include:: snls_storage.rst

functions
---------

   .. function:: snls.initialize()

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
          maxit : int
             the maximum number of iterations performed.
          alive_unit : int
             removal of the file alive_file from unit alive_unit
             terminates execution.
          alive_file : str
             see alive_unit.
          jacobian_available : int
             is the Jacobian matrix of first derivatives available
             ($\geq$ 2), is access only via matrix-vector products
             (=1) or is it not available ($\leq$ 0) ?.
          subproblem_solver : int
             the solver used to compute the step.  Possible values are

             * **1**

               use a projected-gradient method from the GALAHAD module ``slls``.

             * **2**

               use an interior-point method from the GALAHAD module ``sllsb``.

             * **3**

               use an interior-point method, but switch to a 
               projected-gradient method when sufficient progress has 
               been made, see stop_pg_switch below.

          non_monotone : int
             non-monotone <= 0 monotone strategy used, anything else
             non-monotone strategy with this history length used.
          weight_update_strategy : int
             define the weight-update strategy: 1 (basic), 2 (reset to
             zero when very successful), 3 (imitate TR), 4 (increase
             lower bound), 5 (GPT).
          stop_r_absolute : float
             overall convergence tolerances. The iteration will
             terminate when $||r(x)||_W \leq \max($
             ``stop_r_absolute,`` ``stop_r_relative`` *
             $\|r(x_0)\|_W$ or when the norm of the
             gradient, $g(x) = J^T(x) W r(x)$, satisfies
             $\|P[x-g(x)]-x\|_2 \leq \max($ ``stop_pg_absolute,``
             ``stop_pg_relative`` * 
             $\|P[x_0 - g(x_0)] - x_0\|_2$, or if the norm of
             the step is less than ``stop_s``, where $x_0$ is the initial point,
          stop_r_relative : float
             see stop_r_absolute.
          stop_pg_absolute : float
             see stop_r_absolute.
          stop_pg_relative : float
             see stop_r_absolute.
          stop_s : float
             see stop_r_absolute.
          stop_pg_switch : float
             the step-computation solver will switch from an interior-point
             method to a projected-gradient one if subproblem_solver = 3 
             (see above) and $\|P[x-g(x)]-x\|_2 \leq \max($ 
             ``stop_pg_absolute,`` ``stop_pg_switch`` * 
             $\|P[x_0 - g(x_0)] - x_0\|_2$.
          initial_weight : float
             initial value for the regularization weight (-ve means
             $1/\|g_0\|)$).
          minimum_weight : float
             minimum permitted regularization weight.
          eta_successful : float
             potential iterate will only be accepted if the actual
             decrease f - f(x_new) is larger than ``eta_successful``
             times that predicted by a quadratic model of the decrease.
             The regularization weight will be decreaed if this
             relative decrease is greater than ``eta_very_successful``
             but smaller than ``eta_too_successful``.
          eta_very_successful : float
             see eta_successful.
          eta_too_successful : float
             see eta_successful.
          weight_decrease_min : float
             on very successful iterations, the regularization weight
             will be reduced by the factor ``weight_decrease`` but no
             more than ``weight_decrease_min`` while if the iteration
             is unsucceful, the weight will be increased by a factor
             ``weight_increase`` but no more than
             ``weight_increase_max`` (these are delta_1, delta_2,
             delta3 and delta_max in Gould, Porcelli and Toint, 2011).
          weight_decrease : float
             see weight_decrease_min
          weight_increase : float
             see weight_decrease_min
          weight_increase_max : float
             see weight_decrease_min
          switch_to_newton : float
             if newton_acceleration (see below) is true, the Gauss-Newton
             model will switch to the Newton one as soon as the norm of the
             projected gradient is smaller than switch_to_newton.
              **not yet implemented**
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          newton_acceleration : bool
             should available second derivatives be used to accelerate the
             iteration when possible? **not yet implemented**
          magic_step : bool
             allow the user to perform a "magic" step to improve the
             objective.
          print_obj : bool
             print values of the objective/gradient rather than $\|r\|$
             and its gradient.
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
          slls_options : dict
             default control options for `slls` (see ``slls.initialize``).
          sllsb_options : dict
             default control options for `sllsb` (see ``sllsb.initialize``).

   .. function:: snls.load(n, m_r, m_c, J_type, J_ne, J_row, J_col, J_ptr_ne, 
                           J_ptr, cohort, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m_r : int
          holds the number of residuals.
      m_c : int
          holds the number of cohorts.
      J_type : string
          specifies the unsymmetric storage scheme used for the Jacobian
          $J = J(x)$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      Jr_ne : int
          holds the number of entries in $J_r$ in the sparse co-ordinate 
          storage scheme. It need not be set for any of the other schemes.
      Jr_row : ndarray(Jr_ne)
          holds the row indices of $J_r$
          in the sparse co-ordinate storage and sparse by columns schemes
          It need not be set for any of the other schemes, and in this case 
          can be None.
      Jr_col : ndarray(Jr_ne)
          holds the column indices of $J_r$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when the 
          other schemes are used, and in this case can be None.
      Jr_ptr_ne : int
          holds the dimension of J_ptr. This should be at least m_r+1 if $J_r$ 
          is stored in the sparse_by_rows scheme, n+1 for the sparse_by_cols
          scheme and 0 otherwise.
      Jr_ptr : ndarray(J_ptr_ne)
          holds the starting position of each row of $J_r$, as well as the 
          total number of entries, in the sparse row-wise storage 
          scheme, or the starting position of each column of $J_r$, as well 
          as the total number of entries, in the sparse column-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      cohort : ndarray(n), optional
          specifies which cohort, $i=0,\ldots,m_c-1$, variable $x_j$ 
          $j=0,\ldots,n-1$, lies in, or to a negative number if $x_j$
          is not constrained. If all variables lie in a single cohort,
          it may be set to None.
      options : dict, optional
          dictionary of control options (see ``snls.initialize``).

   .. function:: snls.solve(n, m_r, m_c, x, eval_r, Jr_ne, eval_jr, w)

      Find an approximate local unconstrained minimizer of a given 
      least-squares function subject to non-overlapping simplex constraints
      using a regularization method.

      **Parameters:**

      n : int
          holds the number of variables.
      m_r : int
          holds the number of residuals.
      m_c : int
          holds the number of cohorts.
      x : ndarray(n)
          holds the values of optimization variables $x$.
      eval_r : callable
          a user-defined function that must have the signature:

          ``r = eval_r(x)``

          The components of the residual $r(x)$ evaluated at $x$ must be
          assigned to ``r``.
      Jr_ne : int
          holds the number of entries in the Jacobian $J_r = J_r(x)$.
      eval_jr : callable
          a user-defined function that must have the signature:

          ``jr = eval_jr(x)``

          The components of the nonzeros in the Jacobian
          $J_r(x)$ of the objective function evaluated at
          $x$ must be assigned to ``jr`` in the same order as specified
          in the sparsity pattern in ``snls.load``.
      w : ndarray(m_r), optional
          holds the vector of weights $w$. If w is Null, weights of
          one will be presumed.

      **Returns:**

      x : ndarray(n)
          holds the value of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m_c)
          holds the value of the Lagrange multipliers $y$.
      z : ndarray(n)
          holds the value of the dual variables $z$.
      r : ndarray(m_r)
          holds the value of the residuals $r(x)$.
      g : ndarray(n)
          holds the gradient $\nabla f(x)$ of the objective function.


   .. function:: [optional] snls.information()

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

              The restriction n > 0 or m_r > 0 or requirement that type contains
              its relevant string 'dense', 'dense_by_rows', 'dense_by_columns', 
              'sparse_by_rows', 'sparse_by_columns', 'coordinate' or
              'diagonal' has been violated.

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
          inner_iter : int
             the total number of inner iterations performed when finding
             the step.
          r_eval : int
             the total number of evaluations of the residual function
             $r(x)$.
          j_eval : int
             the total number of evaluations of the Jacobian $J_r(x)$ of
             $r(x)$.
          obj : float
             the value of the objective function
             $\frac{1}{2}\|r(x)\|^2_W$ at the best estimate the
             solution, x, determined by SNLS_solve.
          norm_r : float
             the norm of the residual $\|r(x)\|_W$ at the best estimate
             of the solution x, determined by SNLS_solve.
          norm_g : float
             the norm of the gradient of $\|r(x)\|_W$ of the objective
             function at the best estimate, x, of the solution
             determined by SNLS_solve.
          norm_pg : float
             the norm of the projected gradient $P[x- J_r^T(x) r(x)] - x$
             at the best estimate, x, of the solution determined by SNLS_solve.
          weight : float
             the final regularization weight used.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               slls : float
                  the CPU time spent in the ``slls`` package
               sllsb : float
                  the CPU time spent in the ``sllsb`` package
               clock_total : float
                  the total clock time spent in the package.
               clock_slls : float
                  the clock time spent in the ``slls`` package
               clock_sllsb : float
                  the clock time spent in the ``sllsb`` package
          slls_inform : dict
             inform parameters for ``slls`` (see ``slls.information``).
          sllsb_inform : dict
             inform parameters for ``sllsb`` (see ``sllsb.information``).

   .. function:: snls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/snls/Python/test_snls.py
   :code: python

This example code is available in $GALAHAD/src/snls/Python/test_snls.py .
