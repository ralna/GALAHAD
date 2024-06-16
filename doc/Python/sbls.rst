SBLS
====

.. module:: galahad.sbls

.. include:: sbls_intro.rst

.. include:: sbls_storage.rst

functions
---------

   .. function:: sbls.initialize()

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

          indmin : int
             initial estimate of integer workspace for SLS (obsolete).
          valmin : int
             initial estimate of real workspace for SLS (obsolete).
          len_ulsmin : int
             initial estimate of workspace for ULS (obsolete).
          itref_max : int
             maximum number of iterative refinements with
             preconditioner allowed.
          maxit_pcg : int
             maximum number of projected CG iterations allowed.
          new_a : int
             how much has $A$ changed since last factorization.
             Possible values are

             * **0**

               unchanged.

             * **1**

               values but not indices have changed.

             * **2**

               values and indices have changed.


          new_h : int
             how much has $H$ changed since last factorization.
             Possible values are

             * **0**

               unchanged.

             * **1**

               values but not indices have changed.

             * **2**

               values and indices have changed.


          new_c : int
             how much has $C$ changed since last factorization.
             Possible values are

             * **0**

               unchanged.

             * **1**

               values but not indices have changed.

             * **2**

               values and indices have changed.


          preconditioner : int
             which preconditioner to use:

             * **0**

               selected automatically

             * **1**

               explicit with $G = I$

             * **2**

               explicit with $G = H$

             * **3**

               explicit with $G = $ diag(max($H$,min_diag))

             * **4**

               explicit with $G =$ band$(H)$

             * **5**

               explicit with $G =$ (optional, diagonal) $D$

             * **11**

               explicit with $G_{11} = 0$, $G_{21} = 0$,
               $G_{22} = H_{22}$

             * **12**

               explicit with $G_{11} = 0$, $G_{21} = H_{21}$,
               $G_{22} = H_{22}$

             * **-1**

               implicit with $G_{11} = 0$, $G_{21} = 0$,
               $G_{22} = I$

             * **-2**

               implicit with $G_{11} = 0$, $G_{21} = 0$,
               $G_{22} = H_{22}$.

          semi_bandwidth : int
             the semi-bandwidth for band(H).
          factorization : int
             the explicit factorization used:

             * **0**

               selected automatically

             * **1**

               Schur-complement if $G$ is diagonal and successful
               otherwise augmented system

             * **2**

               augmented system

             * **3**

               null-space

             * **4**

               Schur-complement if $G$ is diagonal and successful
               otherwise failure

             * **5**

               Schur-complement with pivoting if $G$ is diagonal and
               successful otherwise failure.

          max_col : int
             maximum number of nonzeros in a column of $A$ for
             Schur-complement factorization.
          scaling : int
             not used at present.
          ordering : int
             see scaling.
          pivot_tol : float
             the relative pivot tolerance used by ULS (obsolete).
          pivot_tol_for_basis : float
             the relative pivot tolerance used by ULS when determining
             the basis matrix.
          zero_pivot : float
             the absolute pivot tolerance used by ULS (obsolete).
          static_tolerance : float
             not used at present.
          static_level : float
             see static_tolerance.
          min_diagonal : float
             the minimum permitted diagonal in
             diag(max($H$,min_diag)).
          stop_absolute : float
             the required absolute and relative accuracies.
          stop_relative : float
             see stop_absolute.
          remove_dependencies : bool
             preprocess equality constraints to remove linear
             dependencies.
          find_basis_by_transpose : bool
             determine implicit factorization preconditioners using a
             basis of A found by examining A's transpose.
          affine : bool
             can the right-hand side $c$ be assumed to be zero?.
          allow_singular : bool
             do we tolerate "singular" preconditioners?.
          perturb_to_make_definite : bool
             if the initial attempt at finding a preconditioner is
             unsuccessful, should the diagonal be perturbed so that a
             second attempt succeeds?.
          get_norm_residual : bool
             compute the residual when applying the preconditioner?.
          check_basis : bool
             if an implicit or null-space preconditioner is used,
             assess and correct for ill conditioned basis matrices.
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
          definite_linear_solver : str
             definite linear equation solver used.
          unsymmetric_linear_solver : str
             unsymmetric linear equation solver used.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          uls_options : dict
             default control options for ULS (see ``uls.initialize``).

   .. function:: sbls.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row, C_col, C_ptr, options=None)

      Import problem data into internal storage prior to solution.

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
          dictionary of control options (see ``sbls.initialize``).

   .. function:: sbls.factorize_matrix(n, m, H_ne, H_val, A_ne, A_val, C_ne, C_val,D)

      Form and factorize the block matrix
      $$K_{G} = \begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix}$$
      for some appropriate matrix $G$.

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
          ``sbls.load``.
      A_ne : int
          holds the number of entries in the matrix $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``sbls.load``.
      C_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $C$.
      C_val : ndarray(C_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $C$ in the same order as specified in the sparsity pattern in 
          ``sbls.load``.
      D : ndarray(n)
          holds the values of the diagonals of the matrix $D$ that is required 
          if options[`preconditioner`]=5 has been specified. Otherwise it
          shuld be set to None.

   .. function:: sbls.solve_system(n, m, rhs)

      Solve the block linear system
      $$\begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix} 
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

   .. function:: [optional] sbls.information()

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

            * **-12**

              The analysis phase of an unsymmetric factorization failed; the 
              return status from the factorization package is given by
              inform['factor_status'].

            * **-13**

              An unsymmetric factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-15**

              The computed preconditioner $P_G$ is singular,
              and is thus unsuitable

            * **-20**

              The computed preconditioner $P_G$ has the wrong inertia, 
              and is thus unsuitable

            * **-24** 

              An error was reported by the sort routine; the return
              status is returned in ``sort_status``.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          sort_status : int
             the return status from the sorting routines.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          preconditioner : int
             the preconditioner used.
          factorization : int
             the factorization used.
          d_plus : int
             how many of the diagonals in the factorization are
             positive.
          rank : int
             the computed rank of $A$.
          rank_def : bool
             is the matrix A rank defficient?.
          perturbed : bool
             has the used preconditioner been perturbed to guarantee
             correct inertia?.
          iter_pcg : int
             the total number of projected CG iterations required.
          norm_residual : float
             the norm of the residual.
          alternative : bool
             has an "alternative" $y$: $K y = 0$ and $y^T c > 0$
             been found when trying to solve $K y = c$ for generic
             $K$?.
          time : dict
             dictionary containing timing information:
               total : float
                  total cpu time spent in the package.
               form : float
                  cpu time spent forming the preconditioner $K_G$.
               factorize : float
                  cpu time spent factorizing $K_G$.
               apply : float
                  cpu time spent solving linear systems inolving $K_G$.
               clock_total : float
                  total clock time spent in the package.
               clock_form : float
                  clock time spent forming the preconditioner $K_G$.
               clock_factorize : float
                  clock time spent factorizing $K_G$.
               clock_apply : float
                  clock time spent solving linear systems inolving $K_G$.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          uls_inform : dict
             inform parameters for ULS (see ``uls.information``).


   .. function:: sbls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/sbls/Python/test_sbls.py
   :code: python

This example code is available in $GALAHAD/src/sbls/Python/test_sbls.py .
