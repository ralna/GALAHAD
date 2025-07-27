SSLS
====

.. module:: galahad.ssls

.. include:: ssls_intro.rst

.. include:: ssls_storage.rst

functions
---------

   .. function:: ssls.initialize()

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

               gives a summary of the progress of the method.

             * **>=2**

               gives increasingly verbose (debugging) output.

          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          symmetric_linear_solver : str
             indefinite linear equation solver used.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).

   .. function:: ssls.analyse(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row, C_col, C_ptr, options=None)

      Assmeble the structure of the block matrix
      $$K = \begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix}.$$
      and analyse the resulting matrix prior to factorization.

      **Parameters:**

      n : int
          holds the dimension of $H$ 
          (equivalently, the number of columns of $A$).
      m : int
          holds the dimension of $C$
          (equivalently, the number of rows of $A$).
      H_type : string
          specifies the symmetric storage scheme used for the matrix $H$.
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
      A_type : string
          specifies the unsymmetric storage scheme used for the matrix $A$,
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
      C_type : string
          specifies the symmetric storage scheme used for the matrix $C$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      C_ne : int
          holds the number of entries in the  lower triangular part of
          $C$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      C_row : ndarray(C_ne)
          holds the row indices of the lower triangular part of $C$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      C_col : ndarray(C_ne)
          holds the column indices of the  lower triangular part of
          $C$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      C_ptr : ndarray(m+1)
          holds the starting position of each row of the lower triangular
          part of $C$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``ssls.initialize``).

   .. function:: ssls.factorize_matrix(n, m, H_ne, H_val, A_ne, A_val, C_ne, C_val)

      Factorize the block matrix
      $$K = \begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix}.$$

      **Parameters:**

      n : int
          holds the dimension of $H$ 
          (equivalently, the number of columns of $A$).
      m : int
          holds the dimension of $C$
          (equivalently, the number of rows of $A$).
      H_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $H$.
      H_val : ndarray(H_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $H$ in the same order as specified in the sparsity pattern in 
          ``ssls.load``.
      A_ne : int
          holds the number of entries in the matrix $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``ssls.load``.
      C_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $C$.
      C_val : ndarray(C_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $C$ in the same order as specified in the sparsity pattern in 
          ``ssls.load``.

   .. function:: ssls.solve_system(n, m, rhs)

      Solve the block linear system
      $$\begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix} 
      \begin{pmatrix}x \\ y\end{pmatrix}= \begin{pmatrix}a \\ b\end{pmatrix}$$


      **Parameters:**

      n : int
          holds the dimension of $H$ 
          (equivalently, the number of columns of $A$).
      m : int
          holds the dimension of $C$
          (equivalently, the number of rows of $A$).

      sol : ndarray(n+m)
          holds the values of the right-hand side vector $(a,b)$.

      **Returns:**

      sol : ndarray(n+m)
          holds the values of the solution vector $(x,y)$ after a successful 
          call.

   .. function:: [optional] ssls.information()

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

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          rank : int
             the computed rank of $A$.
          rank_def : bool
             is the matrix A rank defficient?.
          time : dict
             dictionary containing timing information:
               total : float
                  total cpu time spent in the package.
               analyse : float
                  cpu time spent forming and analysing $K$.
               factorize : float
                  cpu time spent factorizing $K$.
               solve : float
                  cpu time spent solving linear systems inolving $K$.
               clock_total : float
                  total clock time spent in the package.
               clock_analyse : float
                  clock time spent forming and analysing $K$.
               clock_factorize : float
                  clock time spent factorizing $K$.
               clock_solve : float
                  clock time spent solving linear systems inolving $K$.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).

   .. function:: ssls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/ssls/Python/test_ssls.py
   :code: python

This example code is available in $GALAHAD/src/ssls/Python/test_ssls.py .
