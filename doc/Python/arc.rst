ARC
===

.. module:: galahad.arc

.. include:: arc_intro.rst

.. include:: arc_storage.rst

functions
---------

   .. function:: arc.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
            error and warning diagnostics occur on stream error.
          out : int
            general output occurs on stream out.
          print_level : int
            the level of output required. Possible values are:

            * **<= 0**

              no output

            * **1**

              a one-line summary for every improvement

            * **2**

              a summary of each iteration

            * **>= 3**

              increasingly verbose (debugging) output.
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
          more_toraldo : int
            more_toraldo >= 1 gives the number of More'-Toraldo projected
            searches to be used to improve upon the Cauchy point,
            anything else is for the standard add-one-at-a-time CG search.
          non_monotone : int
            non-monotone <= 0 monotone strategy used, anything else
            non-monotone strategy with this history length used.
          model : int
            the model used.  Possible values are

            * **0**

              dynamic (*not yet implemented*)

            * **1**

              first-order (no Hessian)

            * **2**

              second-order (exact Hessian)

            * **3**

              barely second-order (identity Hessian)

            * **4**

              secant second-order (sparsity-based)

            * **5**

              secant second-order (limited-memory BFGS, with``lbfgs_vectors``
              history) (*not yet implemented*)

            * **6**

              secant second-order (limited-memory SR1, with
              ``lbfgs_vectors``  history) (*not yet implemented*).

          norm : int
            The norm is defined via $||v||^2 = v^T P v$, and will define
            the preconditioner used for iterative methods. Possible
            values for $P$ are

            * **-3**

              users own preconditioner

            * **-2**

              $P =$ limited-memory BFGS matrix (with ``lbfgs_vectors`` history)

            * **-1**

              identity (= Euclidan two-norm)

            * **0**

              automatic (*not yet implemented*)

            * **1**

              diagonal, $P =$ diag( max( Hessian, ``min_diagonal`` ) )

            * **2**

              banded, $P =$ band( Hessian ) with semi-bandwidth
              ``semi_bandwidth``

            * **3**

              re-ordered band, P=band(order(A)) with semi-bandwidth
              ``semi_bandwidth``

            * **4**

              full factorization, $P =$ Hessian,  Schnabel-Eskow modification

            * **5**

              full factorization, $P =$ Hessian, GMPS modification
              (*not yet implemented*)

            * **6**

              incomplete factorization of Hessian, Lin-More'

            * **7**

              incomplete factorization of Hessian, HSL_MI28

            * **8**

              incomplete factorization of Hessian, Munskgaard
              (*not yet implemented*)

            * **9**

              expanding band of Hessian (*not yet implemented*).

          semi_bandwidth : int
            specify the semi-bandwidth of the band matrix $P$ if required.
          lbfgs_vectors : int
            number of vectors used by the L-BFGS matrix $P$ if required.
          max_dxg : int
            number of vectors used by the sparsity-based secant Hessian
            if required.
          icfs_vectors : int
            number of vectors used by the Lin-More' incomplete
            factorization matrix $P$ if required.
          mi28_lsize : int
            the maximum number of fill entries within each column of the
            incomplete factor L computed by HSL_MI28. In general,
            increasing ``mi28_lsize`` improve the quality of the
            preconditioner but increases the time to compute and then
            apply the preconditioner. Values less than 0 are treated as 0.
          mi28_rsize : int
            the maximum number of entries within each column of the
            strictly lower triangular matrix $R$ used in the computation
            of the preconditioner by HSL_MI28. Rank-1 arrays of size
            ``mi28_rsize`` * n are allocated internally to hold $R$. Thus
            the amount of memory used, as well as the amount of work
            involved in computing the preconditioner, depends on
            ``mi28_rsize.`` Setting ``mi28_rsize`` > 0 generally leads to
            a higher quality preconditioner than using ``mi28_rsize`` =
            0, and choosing ``mi28_rsize`` >= ``mi28_lsize`` is generally
            recommended.
          advanced_start : int
             try to pick a good initial regularization weight using
             ``advanced_start`` iterates of a variant on the strategy
             of Sartenaer SISC 18(6) 1990:1788-1803.
          stop_g_absolute : float
            overall convergence tolerances. The iteration will terminate
            when the norm of the gradient of the objective function is
            smaller than MAX( ``stop_g_absolute,`` ``stop_g_relative``
            * norm of the initial gradient ) or if the step is less than
            ``stop_s``.
          stop_pg_relative : float
            see stop_g_absolute.
          stop_s : float
            see stop_g_absolute.
          initial_weight : float
             Initial value for the regularisation weight (-ve =>
             1/||g_0||).
          minimum_weight : float
             minimum permitted regularisation weight.
          reduce_gap : float
             expert parameters as suggested in Gould, Porcelli & Toint,
             "Updating the regularization parameter in the adaptive
             cubic regularization algorithm" RAL-TR-2011-007,
             Rutherford Appleton Laboratory, England (2011),
             http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
             (these are denoted beta, epsilon_chi and alpha_max in the
             paper).
          tiny_gap : float
             see reduce_gap.
          large_root : float
             see reduce_gap.
          eta_successful : float
             a potential iterate will only be accepted if the actual
             decrease f - f(x_new) is larger than ``eta_successful``
             times that predicted by a quadratic model of the decrease.
             The regularization weight will be decreased if this
             relative decrease is greater than ``eta_very_successful``
             but smaller than ``eta_too_successful`` (the first is eta
             in Gould, Porcell and Toint, 2011).
          eta_very_successful : float
             see eta_successful.
          eta_too_successful : float
             see eta_successful.
          weight_decrease_min : float
             on very successful iterations, the regularization weight
             will be reduced by the factor ``weight_decrease`` but no
             more than ``weight_decrease_min`` while if the iteration
             is unsuccessful, the weight will be increased by a factor
             ``weight_increase`` but no more than
             ``weight_increase_max`` (these are delta_1, delta_2,
             delta3 and delta_max in Gould, Porcelli and Toint, 2011).
          weight_decrease : float
             see weight_decrease_min.
          weight_increase : float
             see weight_decrease_min.
          weight_increase_max : float
             see weight_decrease_min.
          obj_unbounded : float
            the smallest value the objective function may take before the
            problem is marked as unbounded.
          cpu_time_limit : float
            the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
            the maximum elapsed clock time allowed (-ve means infinite).
          hessian_available : bool
            is the Hessian matrix of second derivatives available or is
            access only via matrix-vector products?.
          subproblem_direct : bool
            use a direct (factorization) or (preconditioned) iterative
            method to find the search direction.
          renormalize_weight : bool
             should the weight be renormalized to account for a change
             in preconditioner?.
          quadratic_ratio_test : bool
             should the test for acceptance involve the quadratic model
             or the cubic?.
          space_critical : bool
            if ``space_critical`` is True, every effort will be made to use
            as little space as possible. This may result in longer
            computation time.
          deallocate_error_fatal : bool
            if ``deallocate_error_fatal`` is True, any array/pointer
            deallocation error will terminate execution. Otherwise,
            computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          trs_options : dict
            default control options for TRS (see ``trs.initialize``).
          gltr_options : dict
            default control options for GLTR (see ``gltr.initialize``).
          dps : dict
            default control options for DPS (see ``dps.initialize``).
          psls_options : dict
            default control options for PSLS (see ``psls.initialize``).
          lms_options : dict
            default control options for LMS (see ``lms.initialize``).
          lms_prec_options : dict
            default control options for LMS (see ``lms.initialize``).
          sec_options : dict
            default control options for SEC (see ``sec.initialize``).
          sha_options : dict
            default control options for SHA (see ``sha.initialize``).

   .. function:: arc.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      H_type : string
          specifies the symmetric storage scheme used for the Hessian.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal' or 'absent', the latter if access to the Hessian
          is via matrix-vector products; lower or upper case variants
          are allowed.
      H_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other three schemes.
      H_row : ndarray(H_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other three schemes, and in this case can be None
      H_col : ndarray(H_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the dense or diagonal
          storage schemes are used, and in this case can be None
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None
      options : dict, optional
          dictionary of control options (see ``arc.initialize``).

   .. function:: arc.solve(n, H_ne, x, eval_f, eval_g, eval_h))

      Find an approximate local unconstrained minimizer of a given function
      using a regularization method.

      **Parameters:**

      n : int
          holds the number of variables.
      H_ne : int
          holds the number of entries in the lower triangular part of $H$.
      x : ndarray(n)
          holds the values of optimization variables $x$.
      eval_f : callable
          a user-defined function that must have the signature:

           ``f = eval_f(x)``

          The value of the objective function $f(x)$
          evaluated at $x$ must be assigned to ``f``.
      eval_g : callable
          a user-defined function that must have the signature:

           ``g = eval_g(x)``

          The components of the gradient $\nabla f(x)$ of the
          objective function evaluated at $x$ must be assigned to ``g``.
      eval_h : callable
          a user-defined function that must have the signature:

           ``h = eval_h(x)``

          The components of the nonzeros in the lower triangle of the Hessian
          $\nabla^2 f(x)$ of the objective function evaluated at
          $x$ must be assigned to ``h`` in the same order as specified
          in the sparsity pattern in ``arc.load``.

      **Returns:**

      x : ndarray(n)
          holds the value of the approximate global minimizer $x$ after
          a successful call.
      g : ndarray(n)
          holds the gradient $\nabla f(x)$ of the objective function.


   .. function:: [optional] arc.information()

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

              The restriction n > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal' or 'absent' has been violated.

            * **-7**

              The objective function appears to be unbounded from below.

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
          iter : int
            the total number of iterations performed.
          cg_iter : int
            the total number of CG iterations performed.
          f_eval : int
            the total number of evaluations of the objective function.
          g_eval : int
            the total number of evaluations of the gradient of the
            objective function.
          h_eval : int
            the total number of evaluations of the Hessian of the
            objective function.
          factorization_max : int
            the maximum number of factorizations in a sub-problem solve.
          factorization_status : int
            the return status from the factorization.
          max_entries_factors : int
            the maximum number of entries in the factors.
          factorization_integer : int
            the total integer workspace required for the factorization.
          factorization_real : int
            the total real workspace required for the factorization.
          obj : float
            the value of the objective function at the best estimate of
            the solution determined by arc.solve.
          norm_g : float
            the norm of the gradient of the objective function
            at the best estimate of the solution determined by arc.solve.
          weight : float
             the current value of the regularization weight.
          time : dict
            dictionary containing timing information:
              total : float
                the total CPU time spent in the package.
              preprocess : float
                the CPU time spent preprocessing the problem.
              analyse : float
                the CPU time spent analysing the required matrices prior to
                factorization.
              factorize : float
                the CPU time spent factorizing the required matrices.
              solve : float
                the CPU time spent computing the search direction.
              clock_total : float
                the total clock time spent in the package.
              clock_preprocess : float
                the clock time spent preprocessing the problem.
              clock_analyse : float
                the clock time spent analysing the required matrices prior to
                factorization.
              clock_factorize : float
                the clock time spent factorizing the required matrices.
              clock_solve : float
                the clock time spent computing the search direction.
          trs_inform : dict
            inform parameters for TRS (see ``trs.information``).
          gltr_inform : dict
            inform parameters for GLTR (see ``gltr.information``).
          dps_inform : dict
            inform parameters for DPS (see ``dps.information``).
          psls_inform : dict
            inform parameters for PSLS (see ``psls.information``).
          lms_inform : dict
            inform parameters for LMS (see ``lms.information``).
          lms_prec_inform : dict
            inform parameters for LMS used for preconditioning
            (see ``lms.information``).
          sec_inform : dict
            inform parameters for SEC (see ``sec.information``).
          sha_inform : dict
            inform parameters for SHA (see ``sha.information``).

   .. function:: arc.terminate()

      Deallocate all internal private storage.

example code
------------

.. include:: ../../src/arc/Python/test_arc.py
   :code: python

This example code is available in $GALAHAD/src/arc/Python/test_arc.py .
