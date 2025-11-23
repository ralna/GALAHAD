TREK
====

.. module:: galahad.trek

.. include:: trek_intro.rst

.. include:: trek_storage.rst

functions
---------

   .. function:: trek.initialize()

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

          eks_max : int
             maximum dimension of the extended Krylov space employed. 
             If a negative value is given, the value 100 will be used instead.
          it_max : int
             the maximum number of iterations allowed.
             If a negative value is given, the value 100 will be used instead.
          f : float
             the value of $f$ in the objective function. This value has no
             effect on the computed $x$, and takes the value 0.0 by default.
          reduction : float
             the value of the reduction factor for a suggested subsequent 
             trust-region radius, see control['next_radius']. The suggested
             radius will be ``reduction`` times the smaller of the current 
             radius and $\|x\|_S$ at the output $x$.
          stop_residual : float
             the value of the stopping tolerance used by the algorithm. The
             iteration stops as soon as $x$ and $\lambda$ are found to satisfy
             $\| ( H + \lambda S ) x + c \| <$ ``stop_residual``
             $\times \max( 1, \|c\| )$.
          reorthogonalize : bool
             should be set to True if the generated basis of the 
             extended-Krylov subspace is to be reorthogonalized at every
             iteration. This can be very expensive, and is generally 
             not warranted.
          s_version_52 : bool
             should be set to True if Algorithm 5.2 in the paper is used
             to generate the extended Krylov space recurrences when a non-unit 
             $S$ is given, and False if those from Algorithm B.3 ares used 
             instead. In practice, there is very little difference in 
             performance and accuracy.
          perturb_c : bool
             should be set to True if the user wishes to make a tiny 
             pseudo-random perturbations to the components of the term $c$ 
             to try to protect from the so-called (probability zero) "hard" 
             case. Perturbations are generally not needed, and should only 
             be used in very exceptional cases.
          stop_check_all_orders : bool
             should be set to True if the algorithm checks for termination
             for each new member of the extended Krylov space. Such checks 
             incur some extra cost, and experience shows that testing every 
             second member is sufficient.
          new_radius : bool
             should be set to True if the call retains the previous $H$, $S$
             and $c$, but with a new, smaller radius.
          new_values : bool
             should be set to True if the any of the values of $H$, $S$
             and $c$ has changed since a previous call.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          linear_solver : str
             linear equation solver used for systems involving $H$.
          linear_solver_for_s : str
             linear equation solver used for systems involving $S$.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          sls_s_options : dict
             default control options for SLS applied to $S$ 
             (see ``sls.initialize``).
          trs_options : dict
             default control options for TRS (see ``trs.initialize``).

   .. function:: trek.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

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
          dictionary of control options (see ``trek.initialize``).

   .. function:: [optional] trek.load_s(n, S_type, S_ne, S_row, S_col, S_ptr, options=None)

      Import problem data for the scaling matrix $S$, if needed, 
      into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      S_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      S_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      S_row : ndarray(S_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      S_col : ndarray(S_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      S_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``trek.initialize``).

   .. function:: [optional] trek.reset_options(options)

      Reset control parameters after import if required.

      **Parameters:**

      options : dict
          dictionary of control options (see ``trek.initialize``).

   .. function:: trek.solve_problem(n, H_ne, H_val, c, radius, S_ne, S_val)

      Find the global minimizer of the quadratic objective function $q(x)$
      within the trust-region constraint.

      **Parameters:**

      n : int
          holds the number of variables.
      H_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(H_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``trek.load``.
      c : ndarray(n)
          holds the values of the linear term $c$ in the objective function.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      S_ne : int
          holds the number of entries in the lower triangular part of 
          the scaling matrix $S$ if it is not the identity matrix. 
          Otherwise it should be None.
      S_val : ndarray(S_ne)
          holds the values of the nonzeros in the lower triangle of the scaling
          matrix $S$ in the same order as specified in the sparsity pattern in 
          ``trek.load_s`` if needed. Otherwise it should be None.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.

   .. function:: [optional] trek.information()

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
              The restriction n > 0, radius > 0,
              or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
              has been violated.

            * **-9**

              The analysis phase of the factorization failed; the return
              status from the factorization package is given by
              inform[‘sls_inform’][‘status’] or 
              inform[‘sls_s_inform’][‘status’] as appropriate.

            * **-10**

              The factorization failed; the return status from the
              factorization package is given by inform[‘sls_inform’][‘status’] 
              or inform[‘sls_s_inform’][‘status’] as appropriate.

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform[‘sls_inform’][‘status’] or 
              inform[‘sls_s_inform’][‘status’] as appropriate.

            * **-15** 

              $S$ does not appear to be strictly diagonally dominant.

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-18**

              Too many iterations have been required. This may happen if 
              options['eks max'] is too small, but may also be symptomatic of 
              a badly scaled problem.

            * **-31** 

              A resolve call has been made before an initial call (see 
              options['new_radius'] and options['new_values']).

            * **-38** 

               An error occurred in a call to an LAPACK subroutine.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required
          n_vec : int
             the number of orthogonal vectors required
          obj : float
             the value of the quadratic function.             
          x_norm : float
             the $S$-norm of $x$, $||x||_S$.
          multiplier : float
             the Lagrange multiplier corresponding to the trust-region
             constraint.
          radius : float
             the current trust-region radius
          next_radius : float
             the  proposed next trust-region radius to be used
          error : float
             the maximum relative residual error
          time : dict
             dictionary containing timing information:
               total : float
                  total CPU time spent in the package.
               assemble : float
                  CPU time spent building $H$ and $S$.
               analyse : float
                  CPU time spent reordering $H$ and $S$ prior to
                  factorization.
               factorize : float
                  CPU time spent factorizing $H$ and $S$.
               solve : float
                  CPU time spent solving linear systems inolving
                  $H$ and $S$.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time spent building $H$ and $S$.
               clock_analyse : float
                  clock time spent reordering $H$ and $S$ prior to
                  factorization.
               clock_factorize : float
                  clock time spent factorizing $H$ and $S$.
               clock_solve : float
                  clock time spent solving linear systems inolving
                  $H$ and $S$.
          sls_inform : dict
             inform parameters for SLS for $H$ (see ``sls.information``).
          sls_s_inform : dict
             inform parameters for SLS for $S$ (see ``sls.information``).
          trs_inform : dict
             inform parameters for TRS (see ``trs.information``).

   .. function:: trek.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/trek/Python/test_trek.py
   :code: python

This example code is available in $GALAHAD/src/trek/Python/test_trek.py .
