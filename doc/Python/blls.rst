BLLS
====

.. module:: galahad.blls

.. include:: blls_intro.rst

.. include:: blls_storage.rst

functions
---------

   .. function:: blls.initialize()

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
             on which iteration to start printing.
          stop_print : int
             on which iteration to stop printing.
          print_gap : int
             how many iterations between printing.
          maxit : int
             how many iterations to perform (-ve reverts to HUGE(1)-1).
          cold_start : int
             cold_start should be set to 0 if a warm start is required
             (with variable assigned according to X_stat, see below),
             and to any other value if the values given in prob.X
             suffice.
          preconditioner : int
             the preconditioner (scaling) used. Possible values are:

             * **0** 

               no preconditioner. 

             * **1** 

               a diagonal preconditioner that normalizes the rows of $A$. 
    
             * **anything else**

               a preconditioner supplied by the user either via  a
               subroutine call of eval_prec} or via reverse
               communication.

          ratio_cg_vs_sd : int
             the ratio of how many iterations use CGLS rather than
             steepest descent.
          change_max : int
             the maximum number of per-iteration changes in the working
             set permitted when allowing CGLS rather than steepest
             descent.
          cg_maxit : int
             how many CG iterations to perform per BLLS iteration (-ve
             reverts to n+1).
          arcsearch_max_steps : int
             the maximum number of steps allowed in a piecewise
             arcsearch (-ve=infini.
          sif_file_device : int
             the unit number to write generated SIF file describing the
             current problem.
          weight : float
             the value of the non-negative regularization weight $\sigma$, 
             i.e., the quadratic objective function $q(x)$ will be regularized 
             by adding $1/2 \sigma \|x\|_2^2$; any value of weight smaller
             than zero will be regarded as zero.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_d : float
             the required accuracy for the dual infeasibility.
          identical_bounds_tol : float
             any pair of constraint bounds $(x_l,x_u)$ that are closer
             than identical_bounds_tol will be reset to the average of
             their values.
          stop_cg_relative : float
             the CG iteration will be stopped as soon as the current
             norm of the preconditioned gradient is smaller than max(
             ``stop_cg_relative`` * initial preconditioned gradient,
             ``stop_cg_absolute``).
          stop_cg_absolute : float
             see ``stop_cg_relative``.
          alpha_max : float
             the largest permitted arc length during the piecewise line
             search.
          alpha_initial : float
             the initial arc length during the inexact piecewise line
             search.
          alpha_reduction : float
             the arc length reduction factor for the inexact piecewise
             line search.
          arcsearch_acceptance_tol : float
             the required relative reduction during the inexact
             piecewise line search.
          stabilisation_weight : float
             the stabilisation weight added to the search-direction
             subproblem.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve = no limit).
          direct_subproblem_solve : bool
             direct_subproblem_solve is True if the least-squares
             subproblem is to be solved using a matrix factorization,
             and False if conjugate gradients are to be preferred.
          exact_arc_search : bool
             exact_arc_search is True if an exact arc_search is
             required, and False if an approximation suffices.
          advance : bool
             advance is True if an inexact exact arc_search can
             increase steps as well as decrease them.
          space_critical : bool
             if space_critical is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation times.
          deallocate_error_fatal : bool
             if deallocate_error_fatal is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if generate_sif_file is True, a SIF file describing the
             current problem will be generated.
          sif_file_name : str
             name (max 30 characters) of generated SIF file containing
             input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          convert_options : dict
             default control options for CONVERT (see ``convert.initialize``).

   .. function:: blls.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          dictionary of control options (see ``blls.initialize``).

   .. function:: blls.solve_ls(n, m, w, a_ne, A_val, b, x_l, x_u, x, z)

      Find a solution to the bound-constraind regularized linear least-squares
      problem involving the least-squares objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      w : ndarray(m)
          holds the weights $w$ in the objective function.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``blls.load``.
      b : ndarray(m)
          holds the values of vector $b$ in the objective function.
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
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] blls.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            return status.  Possible values are:

            * **0**

              The run was succesful.

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

              The bound constraints are inconsistent.

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
             number of iterations required.
          cg_iter : int
             number of CG iterations required.
          obj : float
             current value of the objective function, $r(x)$.
          norm_pg : float
             current value of the Euclidean norm of projected gradient 
             of $r(x)$.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).
          convert_inform : dict
             return information from CONVERT (see ``convert.information``).


   .. function:: blls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/blls/Python/test_blls.py
   :code: python

This example code is available in $GALAHAD/src/blls/Python/test_blls.py .
