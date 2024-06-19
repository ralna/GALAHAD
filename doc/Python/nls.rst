NLS
===

.. module:: galahad.nls

.. include:: nls_intro.rst

.. include:: nls_storage.rst

functions
---------

   .. function:: nls.initialize()

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
          hessian_available : int
             is the Hessian matrix of second derivatives available
             ($\geq$ 2), is access only via matrix-vector products
             (=1) or is it not available ($\leq$ 0) ?.
          model : int
             the model used.  Possible values are

             * **0**

               dynamic (*not yet implemented*)

             * **1**

               first-order (no Hessian)

             * **2**

               barely second-order (identity Hessian)

             * **3**

               Gauss-Newton ($J^T J$ Hessian)

             * **4**

               second-order (exact Hessian)

             * **5**

               Gauss-Newton to Newton transition

             * **6**

               tensor Gauss-Newton treated as a least-squares model

             * **7**

               tensor Gauss-Newton treated as a general model

             * **8**

               tensor Gauss-Newton transition from a least-squares 
               to a general mode.

          norm : int
             the regularization norm used. The norm is defined via
             $\|v\|^2 = v^T S v$, and will define the preconditioner
             used for iterative methods. Possible values for $S$ are

             * **-3**

               user's own regularization norm

             * **-2**

               $S$ = limited-memory BFGS matrix (with
               ``PSLS_options.lbfgs_vectors`` history) (*not yet implemented*)

             * **-1**

               identity (= Euclidan two-norm)

             * **0**

               automatic (*not yet implemented*)

             * **1**

               diagonal, $S$ = diag( max( $J^TJ$ Hessian,
               ``PSLS_options.min_diagonal`` ) )

             * **2**

               diagonal, $S$ = diag( max( Hessian,
               ``PSLS_options.min_diagonal`` ) )

             * **3**

               banded, $S$ = band( Hessian ) with semi-bandwidth
               ``PSLS_options.semi_bandwidth``

             * **4**

               re-ordered band, $S$ = band(order(A)) with semi-bandwidth

               ``PSLS_options.semi_bandwidth``

             * **5**

               full factorization, $S$ = Hessian, Schnabel-Eskow
               modification

             * **6**

               full factorization, $S$ = Hessian, GMPS modification
               (*not yet implemented*)

             * **7**

               incomplete factorization of Hessian, Lin-More'

             * **8**

               incomplete factorization of Hessian, HSL_MI28

             * **9**

               incomplete factorization of Hessian, Munskgaard  
               (*not yet implemented*)

             * **10**

               expanding band of Hessian (*not yet implemented*).

          non_monotone : int
             non-monotone <= 0 monotone strategy used, anything else
             non-monotone strategy with this history length used.
          weight_update_strategy : int
             define the weight-update strategy: 1 (basic), 2 (reset to
             zero when very successful), 3 (imitate TR), 4 (increase
             lower bound), 5 (GPT).
          stop_c_absolute : float
             overall convergence tolerances. The iteration will
             terminate when $||c(x)||_2 \leq \max($
             ``stop_c_absolute,`` ``stop_c_relative`` *
             $\|c(x_{\mbox{initial}})\|_2$ or when the norm of the
             gradient, $g = J^T(x) c(x) / \|c(x)\|_2$, of ||c(x)||_2
             satisfies $\|g\|_2 \leq \max($ ``stop_g_absolute,``
             ``stop_g_relative`` * $\|g_{\mbox{initial}}\|_2$, or if
             the step is less than ``stop_s``.
          stop_c_relative : float
             see stop_c_absolute.
          stop_g_absolute : float
             see stop_c_absolute.
          stop_g_relative : float
             see stop_c_absolute.
          stop_s : float
             see stop_c_absolute.
          power : float
             the regularization power (<2 means chosen according to the
             model).
          initial_weight : float
             initial value for the regularization weight (-ve means
             $1/\|g_0\|)$).
          minimum_weight : float
             minimum permitted regularization weight.
          initial_inner_weight : float
             initial value for the inner regularization weight for
             tensor GN (-ve means 0).
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
          reduce_gap : float
             expert parameters as suggested in Gould, Porcelli and
             Toint, "Updating the regularization parameter in the
             adaptive cubic regularization algorithm" RAL-TR-2011-007,
             Rutherford Appleton Laboratory, England (2011),
             http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
             (these are denoted beta, epsilon_chi and alpha_max in the
             paper).
          tiny_gap : float
             see reduce_gap.
          large_root : float
             see reduce_gap.
          switch_to_newton : float
             if the Gauss-Newto to Newton model is specified, switch to
             Newton as soon as the norm of the gradient g is smaller
             than switch_to_newton.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          subproblem_direct : bool
             use a direct (factorization) or (preconditioned) iterative
             method to find the search direction.
          renormalize_weight : bool
             should the weight be renormalized to account for a change
             in scaling?.
          magic_step : bool
             allow the user to perform a "magic" step to improve the
             objective.
          print_obj : bool
             print values of the objective/gradient rather than ||c||
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
          rqs_options : dict
             default control options for RQS (see ``rqs.initialize``).
          glrt_options : dict
             default control options for GLRT (see ``glrt.initialize``).
          psls_options : dict
             default control options for PSLS (see ``psls.initialize``).
          bsc_options : dict
             default control options for BSC (see ``bsc.initialize``).
          roots_options : dict
             default control options for ROOTS (see ``roots.initialize``).
          subproblem_options : dict
           default control options for the step-finding subproblem:
            error : int
               error and warning diagnostics occur on stream error.
            out : int
               general output occurs on stream out.
            print_level : int
               the level of output required. Possible values are:

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
            hessian_available : int
               is the Hessian matrix of second derivatives available
               ($\geq$ 2), is access only via matrix-vector products
               (=1) or is it not available ($\leq$ 0) ?.
            model : int
               the model used.  Possible values are

               * **0**

                 dynamic (*not yet implemented*)

               * **1**

                 first-order (no Hessian)

               * **2**

                 barely second-order (identity Hessian)

               * **3**

                 Gauss-Newton ($J^T J$ Hessian)

               * **4**

                 second-order (exact Hessian)

               * **5**

                 Gauss-Newton to Newton transition

               * **6**

                 tensor Gauss-Newton treated as a least-squares model

               * **7**

                 tensor Gauss-Newton treated as a general model

               * **8**

                 tensor Gauss-Newton transition from a least-squares 
                 to a general mode.

            norm : int
               the regularization norm used. The norm is defined via
               $\|v\|^2 = v^T S v$, and will define the preconditioner
               used for iterative methods. Possible values for $S$ are

               * **-3**

               user's own regularization norm

               * **-2**

               $S$ = limited-memory BFGS matrix (with

               ``PSLS_options.lbfgs_vectors`` history) (*not yet implemented*)

               * **-1**

               identity (= Euclidan two-norm)

               * **0**

               automatic (*not yet implemented*)

               * **1**

               diagonal, $S$ = diag( max( $J^TJ$ Hessian,
               ``PSLS_options.min_diagonal`` ) )

               * **2**

                 diagonal, $S$ = diag( max( Hessian,
                 ``PSLS_options.min_diagonal`` ) )

               * **3**

                 banded, $S$ = band( Hessian ) with semi-bandwidth
                 ``PSLS_options.semi_bandwidth``

               * **4**

                 re-ordered band, P=band(order(A)) with semi-bandwidth
                 ``PSLS_options.semi_bandwidth``

               * **5**

                 full factorization, $S$ = Hessian, Schnabel-Eskow
                 modification

               * **6**

                 full factorization, $S$ = Hessian, GMPS modification
                 (*not yet implemented*)

               * **7**

                 incomplete factorization of Hessian, Lin-More'

               * **8**

                 incomplete factorization of Hessian, HSL_MI28

               * **9**

                 incomplete factorization of Hessian, Munskgaard  
                 (*not yet implemented*)

               * **10**

                 expanding band of Hessian (*not yet implemented*).

            non_monotone : int
               non-monotone <= 0 monotone strategy used, anything else
               non-monotone strategy with this history length used.
            weight_update_strategy : int
               define the weight-update strategy: 1 (basic), 2 (reset to
               zero when very successful), 3 (imitate TR), 4 (increase
               lower bound), 5 (GPT).
            stop_c_absolute : float
               overall convergence tolerances. The iteration will
               terminate when $||c(x)||_2 \leq \max($
               ``stop_c_absolute,`` ``stop_c_relative`` *
               $\|c(x_{\mbox{initial}})\|_2$ or when the norm of the
               gradient, $g = J^T(x) c(x) / \|c(x)\|_2$, of ||c(x)||_2
               satisfies $\|g\|_2 \leq \max($ ``stop_g_absolute,``
               ``stop_g_relative`` * $\|g_{\mbox{initial}}\|_2$, or if
               the step is less than ``stop_s``.
            stop_c_relative : float
               see stop_c_absolute.
            stop_g_absolute : float
               see stop_c_absolute.
            stop_g_relative : float
               see stop_c_absolute.
            stop_s : float
               see stop_c_absolute.
            power : float
               the regularization power (<2 means chosen according to the
               model).
            initial_weight : float
               initial value for the regularization weight (-ve means
               $1/\|g_0\|)$).
            minimum_weight : float
               minimum permitted regularization weight.
            initial_inner_weight : float
               initial value for the inner regularization weight for
               tensor GN (-ve means 0).
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
            reduce_gap : float
               expert parameters as suggested in Gould, Porcelli and
               Toint, "Updating the regularization parameter in the
               adaptive cubic regularization algorithm" RAL-TR-2011-007,
               Rutherford Appleton Laboratory, England (2011),
               http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
               (these are denoted beta, epsilon_chi and alpha_max in the
               paper).
            tiny_gap : float
               see reduce_gap.
            large_root : float
               see reduce_gap.
            switch_to_newton : float
               if the Gauss-Newto to Newton model is specified, switch to
               Newton as soon as the norm of the gradient g is smaller
               than switch_to_newton.
            cpu_time_limit : float
               the maximum CPU time allowed (-ve means infinite).
            clock_time_limit : float
               the maximum elapsed clock time allowed (-ve means
               infinite).
            subproblem_direct : bool
               use a direct (factorization) or (preconditioned) iterative
               method to find the search direction.
            renormalize_weight : bool
               should the weight be renormalized to account for a change
               in scaling?.
            magic_step : bool
               allow the user to perform a "magic" step to improve the
               objective.
            print_obj : bool
               print values of the objective/gradient rather than ||c||
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
               all output lines will be prefixed by
               ``prefix(2:LEN(TRIM(.prefix))-1)`` where ``prefix``
               contains the required string enclosed in quotes, e.g.
               "string" or 'string'.
            rqs_options : dict
               default control options for RQS (see ``rqs.initialize``).
            glrt_options : dict
               default control options for GLRT (see ``glrt.initialize``).
            psls_options : dict
               default control options for PSLS (see ``psls.initialize``).
            bsc_options : dict
               default control options for BSC (see ``bsc.initialize``).
            roots_options : dict
               default control options for ROOTS (see ``roots.initialize``).

   .. function:: nls.load(n, m, J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne,                          H_row, H_col, H_ptr, P_type, P_ne, P_row, P_col, P_ptr, w, options=None)

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
          $H = H(x,y)$.
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
      P_type : string, optional
          specifies the unsymmetric storage scheme used for the residual
          Hessian-product matrix $P = P(x,v)$.
          It should be one of 'sparse_by_columns' or 'dense_by_columns'
          (with the intention that 'coordinate' will be added at some time);;
          lower or upper case variants are allowed.
          This and the following P_* arguments are only required if
          a tensor Gauss-Newton approximation model is required 
          (see control.model = 6,7,8).
      P_ne : int, optional
          holds the number of entries in $P$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other two schemes.
      P_row : ndarray(P_ne), optional
          holds the row indices of $P$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other two schemes, and in this case can be None.
      P_col : ndarray(P_ne), optional
          holds the column indices of $P$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when the 
          dense storage scheme is used, and in this case can be None.
      P_ptr : ndarray(n+1), optional
          holds the starting position of each column of $P$, as well as the 
          total number of entries, in the sparse column-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      w : ndarray(m), optional
          holds the vector of weights $w$. If w is not provided, weights of
          one will be presumed.
      options : dict, optional
          dictionary of control options (see ``nls.initialize``).

   .. function:: nls.solve(n, m, x, eval_c, J_ne, eval_j, H_ne, eval_h, P_ne, eval_hprod)

      Find an approximate local unconstrained minimizer of a given 
      least-squares function using a regularization method.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      x : ndarray(n)
          holds the values of optimization variables $x$.
      eval_c : callable
          a user-defined function that must have the signature:

          ``c = eval_c(x)``

          The components of the residual $c(x)$ evaluated at $x$ must be
          assigned to ``c``.
      J_ne : int
          holds the number of entries in the Jacobian $J = J(x)$.
      eval_j : callable
          a user-defined function that must have the signature:

          ``j = eval_(x)``

          The components of the nonzeros in the Jacobian
          $J(x)$ of the objective function evaluated at
          $x$ must be assigned to ``j`` in the same order as specified
          in the sparsity pattern in ``nls.load``.
      H_ne : int, optional
          holds the number of entries in the lower triangular part of 
          the Hessian $H = H(x,y$.
          This and the following eval_h argument are only required if
          a Newton approximation or tensor Gauss-Newton approximation
          model is required (see control.model = 4,...,8).
      eval_h : callable, optional
          a user-defined function that must have the signature:

          ``h = eval_h(x,y)``

          The components of the nonzeros in the lower triangle of the Hessian
          $H(x,y)$ evaluated at $x$ and $y$ must be assigned to ``h`` in the 
          same order as specified in the sparsity pattern in ``nls.load``.
      P_ne : int, optional
          holds the number of entries in the residual
          Hessian-product matrix $P = P(x,v)$.
          This and the following eval_hprod argument are only required if
          a Newton approximation or tensor Gauss-Newton approximation
          model is required (see control.model = 6,7,8).
      eval_hprod : callable, optional
          a user-defined function that must have the signature:

          ``p = eval_hprod(x,v)``

          The components of the nonzeros in the Hessian producr matrix
          $P(x,y)$ evaluated at $x$ and $v$ must be assigned to ``p`` in the 
          same order as specified in the sparsity pattern in ``nls.load``.

      **Returns:**

      x : ndarray(n)
          holds the value of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the value of the residuals $c(x)$.
      g : ndarray(n)
          holds the gradient $\nabla f(x)$ of the objective function.


   .. function:: [optional] nls.information()

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
          cg_iter : int
             the total number of CG iterations performed.
          c_eval : int
             the total number of evaluations of the residual function
             c(x).
          j_eval : int
             the total number of evaluations of the Jacobian J(x) of
             c(x).
          h_eval : int
             the total number of evaluations of the scaled Hessian
             H(x,y) of c(x).
          factorization_max : int
             the maximum number of factorizations in a sub-problem
             solve.
          factorization_status : int
             the return status from the factorization.
          max_entries_factors : long
             the maximum number of entries in the factors.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          factorization_average : float
             the average number of factorizations per sub-problem solve.
          obj : float
             the value of the objective function
             $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the
             solution, x, determined by NLS_solve.
          norm_c : float
             the norm of the residual $\|c(x)\|_W$ at the best estimate
             of the solution x, determined by NLS_solve.
          norm_g : float
             the norm of the gradient of $\|c(x)\|_W$ of the objective
             function at the best estimate, x, of the solution
             determined by NLS_solve.
          weight : float
             the final regularization weight used.
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
          rqs_inform : dict
             inform parameters for RQS (see ``rqs.information``).
          glrt_inform : dict
             inform parameters for GLTR (see ``glrt.information``).
          psls_inform : dict
             inform parameters for PSLS (see ``psls.information``).
          bsc_inform : dict
             inform parameters for BSC (see ``bsc.information``).
          roots_inform : dict
             inform parameters for ROOTS (see ``roots.information``).
          subproblem_inform : dict
           inform parameters for subproblem:
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
            cg_iter : int
               the total number of CG iterations performed.
            c_eval : int
               the total number of evaluations of the residual function
               c(x).
            j_eval : int
               the total number of evaluations of the Jacobian J(x) of
               c(x).
            h_eval : int
               the total number of evaluations of the scaled Hessian
               H(x,y) of c(x).
            factorization_max : int
               the maximum number of factorizations in a sub-problem
               solve.
            factorization_status : int
               the return status from the factorization.
            max_entries_factors : long
               the maximum number of entries in the factors.
            factorization_integer : long
               the total integer workspace required for the factorization.
            factorization_real : long
               the total real workspace required for the factorization.
            factorization_average : float
               the average number of factorizations per sub-problem solve.
            obj : float
               the value of the objective function
               $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the
               solution, x, determined by NLS_solve.
            norm_c : float
               the norm of the residual $\|c(x)\|_W$ at the best estimate
               of the solution x, determined by NLS_solve.
            norm_g : float
               the norm of the gradient of $\|c(x)\|_W$ of the objective
               function at the best estimate, x, of the solution
               determined by NLS_solve.
            weight : float
               the final regularization weight used.
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
            rqs_inform : dict
               inform parameters for RQS (see ``rqs.information``).
            glrt_inform : dict
               inform parameters for GLTR (see ``glrt.information``).
            psls_inform : dict
               inform parameters for PSLS (see ``psls.information``).
            bsc_inform : dict
               inform parameters for BSC (see ``bsc.information``).
            roots_inform : dict
               inform parameters for ROOTS (see ``roots.information``).

   .. function:: nls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/nls/Python/test_nls.py
   :code: python

This example code is available in $GALAHAD/src/nls/Python/test_nls.py .
