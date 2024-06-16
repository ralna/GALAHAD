LPA
===

.. module:: galahad.lpa

.. include:: lpa_intro.rst

.. include:: lpa_storage.rst

functions
---------

   .. function:: lpa.initialize()

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

               gives a summary.

             * **2**

               gives a summary of the inner iteration for each iteration
               by enabling output from LA04.

             * **>=3**

               gives increasingly verbose (debugging) output.

          print_level : int
             the level of output required is specified by print_level
             (>= 2 turns on LA04 output).
          start_print : int
             any printing will start on this iteration.
          stop_print : int
             any printing will stop on this iteration.
          maxit : int
             at most maxit inner iterations are allowed.
          max_iterative_refinements : int
             maximum number of iterative refinements allowed.
          min_real_factor_size : int
             initial size for real array for the factors and other data.
          min_integer_factor_size : int
             initial size for integer array for the factors and other
             data.
          random_number_seed : int
             the initial seed used when generating random numbers.
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          qplib_file_device : int
             specifies the unit number to write generated QPLIB file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          tol_data : float
             the tolerable relative perturbation of the data ($A$,$g,\ldots$)
             defining the problem.
          feas_tol : float
             any constraint violated by less than feas_tol will be
             considered to be satisfied.
          relative_pivot_tolerance : float
             pivot threshold used to control the selection of pivot
             elements in the matrix factorization. Any potential pivot
             which is less than the largest entry in its row times the
             threshold is excluded as a candidate.
          growth_limit : float
             limit to control growth in the upated basis factors. A
             refactorization occurs if the growth exceeds this limit.
          zero_tolerance : float
             any entry in the basis smaller than this is considered
             zero.
          change_tolerance : float
             any solution component whose change is smaller than a
             tolerence times the largest change may be considered to be
             zero.
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer than identical_bounds_tol will be reset to the
             average of their values.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          scale : bool
             if ``scale`` is True, the problem will be automatically
             scaled prior to solution. This may improve computation
             time and accuracy.
          dual : bool
             should the dual problem be solved rather than the primal?.
          warm_start : bool
             should a warm start using the data in C_stat and X_stat be
             attempted?.
          steepest_edge : bool
             should steepest-edge weights be used to detetrmine the
             variable leaving the basis?.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made
             to use as little space as possible. This may result in
             longer computation time.
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

   .. function:: lpa.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          dictionary of control options (see ``lpa.initialize``).

   .. function:: lpa.solve_lp(n, m, f, g, A_ne, A_val, c_l, c_u, x_l, x_u)

      Find a solution to the convex quadratic program involving the
      quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      A_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``lpa.load``.
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
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] lpa.information()

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

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          la04_job : int
             the final value of LA04's job argument.
          la04_job_info : int
             any extra information from an unsuccessfull call to LA04
             (LA04's RINFO(35).
          obj : float
             the value of the objective function at the best estimate
             of the solution.
          primal_infeasibility : float
             the value of the primal infeasibility.
          feasible : bool
             is the returned "solution" feasible?.
          RINFO : ndarray(40)
             the information array from LA04.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
          rpd_inform : dict
             inform parameters for RPD (see ``rpd.information``).


   .. function:: lpa.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/lpa/Python/test_lpa.py
   :code: python

This example code is available in $GALAHAD/src/lpa/Python/test_lpa.py .
