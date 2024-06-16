LLST
====

.. module:: galahad.llst

.. include:: llst_intro.rst

.. include:: llst_storage.rst

functions
---------

   .. function:: llst.initialize()

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
          new_a : int
             how much of $A$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1** 

               values but not indices have changed

             * **2** 

               values and indices have changed.
          new_s : int
             how much of $S$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1** 

               values but not indices have changed

             * **2** 

               values and indices have changed.
          max_factorizations : int
             the maximum number of factorizations (=iterations)
             allowed. -ve implies no limit.
          taylor_max_degree : int
             maximum degree of Taylor approximant allowed (<= 3).
          initial_multiplier : float
             initial estimate of the Lagrange multipler.
          lower : float
             lower and upper bounds on the multiplier, if known.
          upper : float
             see lower.
          stop_normal : float
             stop when $| \|x\| -\Delta| \leq$ 
             $\max($ stop_normal * $\max( 1, \Delta )$.
          equality_problem : bool
             is the solution is <b<required</b> to lie on the boundary
             (i.e., is the constraint an equality)?.
          use_initial_multiplier : bool
             ignore initial_multiplier?.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          definite_linear_solver : str
             name of the definite linear equation solver employed.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sbls_control : dict
             control parameters for the symmetric factorization and
             related linear solves (see ``sbls.initialize``).
          sls_control : dict
             control parameters for the factorization of $S$ and related
             linear solves (see ``sls.initialize``).
          ir_control : dict
             control parameters for iterative refinement for definite
             system solves (see ``ir.initialize``).

   .. function:: llst.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      m : int
          holds the number of observations, $m$ (= the number of rows of $A$).
      n : int
          holds the number of variables, $n$ (= the number of columns of $A$).
      A_type : string
          specifies the unsymmetric storage scheme used for the model 
          matrix $A$.
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
          dictionary of control options (see ``llst.initialize``).

   .. function:: [optional] llst.load_scaling(n, S_type, S_ne, S_row, S_col, S_ptr, options=None)

      Import non-trivial trust-region scaling data into internal storage 
      prior to solution. This is only required if $S$ is not the identity
      matrix $I$.

      **Parameters:**

      n : int
          holds the number of variables, $n$ 
          (= the number of rows/columns of $S$).
      S_type : string
          specifies the symmetric storage scheme used for the scaling matrix 
          $S$. It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      S_ne : int
          holds the number of entries in the  lower triangular part of
          $S$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      S_row : ndarray(S_ne)
          holds the row indices of the lower triangular part of $S$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      S_col : ndarray(S_ne)
          holds the column indices of the  lower triangular part of
          $S$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      S_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $S$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``llst.initialize``).

   .. function:: llst.solve_problem(m, n, radius, A_ne, A_val, b, S_ne, S_val)

      Solve the linear-least-squares trust-region problem.

      **Parameters:**

      m : int
          holds the number of observations, $m$.
      n : int
          holds the number of variables, $n$.
      radius : float
          holds the strictly positive trust-region radius, $\Delta$.
      A_ne : int
          holds the number of entries in the model matrix $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in $A$ in the same order as 
          specified in the sparsity pattern in ``llst.load``.
      b : ndarray(m)
          holds the values of the observations $b$.
      S_ne : int, optional
          holds the number of entries in the lower triangular part of 
          the scaling matrix $S$ if it is not the identity matrix. 
          Otherwise it should be None.
      S_val : ndarray(S_ne), optional
          holds the values of the nonzeros in the lower triangle of $S$ in 
          the same order as specified in the sparsity pattern in 
          ``llst.load_scaling`` if needed. Otherwise it should be None.

      **Returns:**

      x : ndarray(n)
          holds the values of the minimizer $x$ after a successful call.

   .. function:: [optional] llst.information()

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

            * **-9**

              The analysis phase of the factorization of $K(\lambda)$ failed; 
              the return status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              The factorization of $K(\lambda)$ failed; the return status from
              the factorization package is given by inform['factor_status'].

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform['factor_status'].

            * **-15**

              The Hessian $S$ appears not to be strictly diagonally
              dominant.

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-17**

              The step is too small to make further progress.

            * **-23** 

              An entry from the strict upper triangle of $S$ has been specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          factorizations : int
             the number of factorizations performed.
          len_history : int
             the number of $(\lambda,\|x\|_S,\|Ax-b\|)$ triples in the history.
          multiplier : float
             the Lagrange multiplier corresponding to the trust-region
             constraint.
          x_norm : float
             the S-norm of x, $\|x\|_S$.
          r_norm : float
             corresponding value of the two-norm of the residual,
             $\|A x(\lambda) - b\|$.
          time : dict
             dictionary containing timing information:
               total : float
                  total CPU time spent in the package.
               assemble : float
                  CPU time assembling $K(\lambda)$ in (1).
               analyse : float
                  CPU time spent analysing $K(\lambda)$.
               factorize : float
                  CPU time spent factorizing $K(\lambda)$.
               solve : float
                  CPU time spent solving linear systems inolving
                  $K(\lambda)$.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time assembling $K(\lambda)$.
               clock_analyse : float
                  clock time spent analysing $K(\lambda)$.
               clock_factorize : float
                  clock time spent factorizing $K(\lambda)$.
               clock_solve : float
                  clock time spent solving linear systems inolving
                  $K(\lambda)$.
          history : dict
             dictionary recording the history of the iterates:
               lambda : ndarray(100)
                  the values of $\lambda$ for the first min(100,
                  ``len_history``) iterations.
               x_norm : ndarray(100)
                  the corresponding values of $\|x(\lambda)\|_S$.
               r_norm : ndarray(100)
                  the corresponding values of $\|A x(\lambda) - b\|_2$.
          sbls_inform : dict
             information from the symmetric factorization and related
             linear solves (see ``sbls.information``).
          sls_inform : dict
             information from the factorization of S and related linear
             solves (see ``sls.information``).
          ir_inform : dict
             information from the iterative refinement for definite
             system solves (see ``ir.information``).

   .. function:: llst.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/llst/Python/test_llst.py
   :code: python

This example code is available in $GALAHAD/src/llst/Python/test_llst.py .
