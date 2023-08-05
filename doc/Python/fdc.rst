FDC
===

.. module:: galahad.fdc

.. include:: fdc_intro.rst

.. include:: fdc_storage.rst

functions
---------

   .. function:: fdc.initialize()

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

          indmin : int
             initial estimate of integer workspace for sls (obsolete).
          valmin : int
             initial estimate of real workspace for sls (obsolete).
          pivot_tol : float
             the relative pivot tolerance (obsolete).
          zero_pivot : float
             the absolute pivot tolerance used (obsolete).
          max_infeas : float
             the largest permitted residual.
          use_sls : bool
             choose whether ``sls`` or ``uls`` is used to determine
             dependencies.
          scale : bool
             should the rows of A be scaled to have unit infinity norm
             or should no scaling be applied?
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          symmetric_linear_solver : str
             symmetric (indefinite) linear equation solver. 
             For current choices, see ``sls.initialize``.
          unsymmetric_linear_solver : str
             unsymmetric linear equation solver.
             For current choices, see ``uls.initialize``.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          uls_options : dict
             default control options for ULS (see ``uls.initialize``).

   .. function:: fdc.find_dependent_rows(m, n, A_ptr, A_col, A_val, b, options=None)

      Find dependent rows of $A$ and, if any, check if $Ax = b$ is consistent.

      **Parameters:**

      m : int
          holds the number of constraints (rows of $A$).
      n : int
          holds the number of variables (columns of $A$).
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries.
      A_col : ndarray(A_ptr(m)-1)
          holds the column indices of the nonzeros of $A$ in the sparse 
          co-ordinate  storage scheme.
      A_val : ndarray(A_ptr(m)-1)
          holds the values of the nonzeros of $A$ in the sparse co-ordinate
          storage scheme.
      b : ndarray(m)
          holds the values of the llinear term $b$ in the constraints.
      options : dict, optional
          dictionary of control options (see ``fdc.initialize``).

      **Returns:**

      m_depen : int
          holds the number of dependent constraints, if any.
      depen : ndarray(m)
          the first m_depen components hold the indices of the dependent
          comstraints.
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

            * **-5**

              The constraints appear to be inconsistent.

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
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linear dependent constraints.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               clock_total : float
                  the total clock time spent in the package.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorization.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          uls_inform : dict
             inform parameters for ULS (see ``uls.information``).

   .. function:: fdc.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/fdc/Python/test_fdc.py
   :code: python

This example code is available in $GALAHAD/src/fdc/Python/test_fdc.py .
