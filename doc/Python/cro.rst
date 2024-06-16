CRO
===

.. module:: galahad.cro

.. include:: cro_intro.rst

.. include:: cro_storage.rst

functions
---------

   .. function:: cro.initialize()

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

               gives no output.

             * **1**

               a summary of the progress made.

             * **>=2**

               an ever increasing amount of debugging information.

          max_schur_complement : int
             the maximum permitted size of the Schur complement before
             a refactorization is performed.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          feasibility_tolerance : float
             feasibility tolerance for KKT violation.
          check_io : bool
             if ``check_io`` is True, the input (x,y,z) will be fully
             tested for consistency.
          refine_solution : bool
             if ``refine`` solution is True, attempt to satisfy the KKT
             conditions as accurately as possible.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made
             to use as little space as possible. This may result in
             longer computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          symmetric_linear_solver : str
             indefinite linear equation solver.
          unsymmetric_linear_solver : str
             unsymmetric linear equation solver.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          uls_options : dict
             default control options for ULS (see ``uls.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          ir_options : dict
             default control options for IR (see ``ir.initialize``).

   .. function:: cro.crossover_solution(n, m, m_equal, g, H_ne, H_val, H_col, H_ptr, A_ne, A_val, A_col, A_ptr, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat, options=None)

      Crossover a primal-dual interior-point solution to a basic one.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
      m_equal : int
          holds the number of equality constraints. These **must** occur 
          first in $A$.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      H_ne : int
          holds the number of entries in the lower triangular part of $H$.
      H_val : ndarray(H_ptr(n)-1)
          holds the values of the nonzeros of the lower triangular 
          part of $H$ in the sparse row-wise storage scheme.
      H_col : ndarray(H_ptr(n)-1)
          holds the column indices of the nonzeros of the lower triangular 
          part of $H$ in the sparse row-wise storage scheme.
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries.
      A_ne : int
          holds the number of entries in $A$.
      A_val : ndarray(A_ptr(m)-1)
          holds the values of the nonzeros of $A$ in the sparse row-wise
          storage scheme.
      A_col : ndarray(A_ptr(m)-1)
          holds the column indices  of the nonzeros of $A$ in the sparse 
          row-wise storage scheme.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries.
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
          holds the values of the approximate minimizer $x$.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      x_stat : ndarray(n)
          holds the input status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.
      c_stat : ndarray(m)
          holds the input status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      options : dict, optional
          dictionary of control options (see ``cro.initialize``).

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be -1 if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound and the constraint is basic and active, -2 
          if it is non-basic and active, 1 if it lies on its upper bound 
          and is active, 2 if it non-basic and active, and
          0 if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will 
          be -1 if the $i$-th variable lies on 
          its lower bound and the variable is basic and active, -2 
          if it is non-basic and active, 1 if it lies on its upper bound 
          and is active, 2 if it non-basic and active, and
          0 if it lies between bounds.
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

              The restriction n > 0 or m > 0 or 0 <= m_equal <= m 
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

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform['factor_status'].

            * **-12**

              The analysis phase of an unsymmetric factorization failed; the 
              return status from the factorization package is given by
              inform['factor_status'].

            * **-13**

              An unsymmetric factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-14**

              The solution of a set of linear equations using factors
              from an unsymmetric factorization package failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-16**

              The resuduals are large, the factorization may be unsatisfactory.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          dependent : int
             the number of dependent active constraints.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               analyse : float
                  the CPU time spent reordering the matrix prior to
                  factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing corrections.
               clock_total : float
                  the total clock time spent in the package.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorizat.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing corrections.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          uls_inform : dict
             inform parameters for ULS (see ``uls.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).
          scu_status : int
             status value for SCU (see ``scu.status``).
          scu_inform : dict
             inform parameters for SCU (see ``scu.information``).
          ir_inform : dict
             return information from IR (see ``ir.information``).


   .. function:: cro.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/cro/Python/test_cro.py
   :code: python

This example code is available in $GALAHAD/src/cro/Python/test_cro.py .
