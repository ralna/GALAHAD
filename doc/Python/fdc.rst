FDC
===

.. module:: galahad.fdc

Given an under-determined set of linear equations/constraints $a_i^T x =
b_i^{}$, $i = 1, \ldots, m$ involving $n \geq m$ unknowns $x$, the ``fdc``
package **determines whether the constraints are consistent, and if
so how many of the constraints are dependent**; a list of dependent
constraints, that is, those which may be removed without changing the
solution set, will be found and the remaining $a_i$ will be linearly
independent.  Full advantage is taken of any zero coefficients in the
matrix $A$ whose columns are the vectors $a_i^T$.

See Section 4 of $GALAHAD/doc/fdc.pdf for a brief description of the
method employed and other details.

matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ must be presented
and stored in *sparse row-wise storage* format.
For this, only the nonzero entries are stored, and they are
ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries plus one.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.

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
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
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
          sls_control : dict
             control parameters for SLS (see ``sls.initialize``).
          uls_control : dict
             control parameters for ULS (see ``uls.initialize``).

   .. function:: fdc.find_dependent_rows(m, n, A_val, A_col, A_ptr, b, options=None)

      Find dependent rows of $A$ and, if any, check if $Ax = b$ is consistent.

      **Parameters:**

      m : int
          holds the number of constraints (rows of $A$).
      n : int
          holds the number of variables (columns of $A$).
      A_val : ndarray(A_ptr(m)-1)
          holds the values of the nonzeros of $A$ in the sparse co-ordinate
          storage scheme.
      A_col : ndarray(A_ptr(m)-1)
          holds the column indices of the nonzeros of $A$ in the sparse 
          co-ordinate  storage scheme.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries plus one.
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

              The run was succesful.

            * **-1**

              An allocation error occurred. A message indicating the
              offending array is written on unit control['error'], and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-2**

              A deallocation error occurred.  A message indicating the
              offending array is written on unit control['error'] and
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
             error ocurred.
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
