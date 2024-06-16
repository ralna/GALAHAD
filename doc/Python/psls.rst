PSLS
====

.. module:: galahad.psls

.. include:: psls_intro.rst

.. include:: psls_storage.rst

functions
---------

   .. function:: psls.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          warning : int
             unit for warning messages.
          out : int
             general output occurs on stream out.
          statistics : int
             unit for statistical output.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output.

             * **1**

               gives a summary of the process.

             * **>=2**

               gives increasingly verbose (debugging) output.

          preconditioner : int
             which preconditioner to use. Possible values are:

             * **<0**

               no preconditioning occurs, $P = I$.

             * **0**

               the preconditioner is chosen automatically
               (forthcoming, and currently defaults to 1).

             * **1**

               $A$ is replaced by the diagonal,  $P$ = diag( max(
               $A$, ``min_diagonal`` ) ).

             * **2**

               $A$ is replaced by the band  $P$ = band( $A$ ) with
               semi-bandwidth ``semi_bandwidth.``

             * **3**

               $A$ is replaced by the reordered band  $P$ = band(
               order( $A$ ) ) with semi-bandwidth  ``semi_bandwidth,``
               where order is chosen by the HSL package  MC61 to move
               entries closer to the diagonal.

             * **4**

               $P$ is a full factorization of $A$ using
               Schnabel-Eskow  modifications, in which small or negative
               diagonals are  made sensibly positive during the factorization.

             * **5**

               $P$ is a full factorization of $A$ due to Gill,
               Murray,  Ponceleon and Saunders, in which an indefinite
               factorization  is altered to give a positive definite one.

             * **6**

               $P$ is an incomplete Cholesky factorization of $A$
               using  the package ICFS due to Lin and More'.

             * **7**

               $P$ is an incomplete factorization of $A$ implemented
               as HSL_MI28 from HSL.

             * **8**

               $P$ is an incomplete factorization of $A$ due  to
               Munskgaard (forthcoming).

             * **>8**

               treated as 0.  

             Options 3-8 may require
             additional external software that is not part of the
             package, and that must be obtained separately.
          semi_bandwidth : int
             the semi-bandwidth for band(H) when ``preconditioner`` =
             2,3.
          scaling : int
             not used at present.
          ordering : int
             see scaling.
          max_col : int
             maximum number of nonzeros in a column of $A$ for
             Schur-complement factorization to accommodate newly
             deleted rpws and columns.
          icfs_vectors : int
             number of extra vectors of length n required by the
             Lin-More' incomplete Cholesky preconditioner when
             ``preconditioner`` = 6.
          mi28_lsize : int
             the maximum number of fill entries within each column of
             the incomplete factor L computed by HSL_MI28 when
             ``preconditioner`` = 7. In general, increasing mi28_lsize
             improve the quality of the preconditioner but increases
             the time to compute and then apply the preconditioner.
             Values less than 0 are treated as 0.
          mi28_rsize : int
             the maximum number of entries within each column of the
             strictly lower triangular matrix $R$ used in the
             computation of the preconditioner by HSL_MI28 when
             ``preconditioner`` = 7. Rank-1 arrays of size
             ``mi28_rsize`` * n are allocated internally to hold $R$.
             Thus the amount of memory used, as well as the amount of
             work involved in computing the preconditioner, depends on
             mi28_rsize. Setting mi28_rsize > 0 generally leads to a
             higher quality preconditioner than using mi28_rsize = 0,
             and choosing mi28_rsize >= mi28_lsize is generally
             recommended.
          min_diagonal : float
             the minimum permitted diagonal in
             diag(max(H,.min_diagonal)).
          new_structure : bool
             set new_structure True if the storage structure for the
             input matrix has changed, and False if only the values
             have changed.
          get_semi_bandwidth : bool
             set get_semi_bandwidth True if the semi-bandwidth of the
             submatrix is to be calculated.
          get_norm_residual : bool
             set get_norm_residual True if the residual when applying
             the preconditioner are to be calculated.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          definite_linear_solver : str
             the definite linear equation solver used when
             ``preconditioner`` = 3,4. Possible choices are currently:
             sils, ma27, ma57, ma77, ma86, ma87, ma97, ssids, mumps, pardiso,
             mkl_pardiso, pastix, wsmp, potr and pbtr, although only sils,
             potr, pbtr and, for OMP 4.0-compliant compilers, ssids are
             installed by default.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          mi28_options : dict
             default control options for HSL_MI28 (see ``mi28.initialize``).

   .. function:: psls.load(n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to factorization.

      **Parameters:**

      n : int
          holds the dimension of the system, $n$.
      A_type : string
          specifies the symmetric storage scheme used for the matrix $A$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in the  lower triangular part of
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      A_row : ndarray(A_ne)
          holds the row indices of the lower triangular part of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of the  lower triangular part of
          $A$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      A_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $A$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``psls.initialize``).

   .. function:: psls.form_preconditioner(A_ne, A_val)

      Form and factorize the preconditioner $P$ from the matrix $A$.

      **Parameters:**

      A_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``psls.load``.

   .. function:: psls.form_subset_preconditioner(a_ne, A_val,n_sub,sub)

      Form and factorize the preconditioner of a symmetric subset of the
      rows and columns of $A$.

      **Parameters:**

      A_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(A_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $A$ in the same order as specified in the sparsity pattern in 
      n_sub : int
          holds the number of rows (and columns) of the required submatrix 
          of $A$.
      sub : ndarray(n_sub)
          holds the indices of the rows of the required submatrix of $A$.

   .. function:: psls.update_preconditioner(a_ne, A_val,n_del,del)

      Update the preconditioner $P$ when rows (and columns) are removed.

      **Parameters:**

      a_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $A$ in the same order as specified in the sparsity pattern in 
      n_del : int
          holds the number of rows (and columns) that will be removed from $A$.
      del : ndarray(n_del)
          holds the indices of the rows that will be removed from $A$.

   .. function:: psls.apply_preconditioner(n, b)

      Solve the system $Px=b$

      **Parameters:**

      n : int
          holds the dimension of the system, $n$.
          holds the number of variables.
      b : ndarray(n)
          holds the values of the right-hand side vector $b$.
          Any component corresponding to rows/columns not in the initial 
          subset recorded by ``psls.form_subset_preconditioner``, or
          in those subsequently deleted by ``psls_update_preconditioner``,
          will not be altered.

      **Returns:**

      x : ndarray(n)
          holds the values of the solution $x$ after a successful call.
          Any component corresponding to rows/columns not in the initial 
          subset recorded by ``psls.form_subset_preconditioner``, or
          in those subsequently deleted by ``psls_update_preconditioner``,
          will be zero.

   .. function:: [optional] psls.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             reported return status. Possible values are

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
               'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
               has been violated.

             * **-9**

               The analysis phase of the factorization failed; the return
               status from the factorization package is given by
               inform['factor_status'].

             * **-10**
 
               The factorization failed; the return status from the
               factorization package is given by inform['factor_status'].

             * **-20**

               The matrix $A$ is not positive definite while the factorization 
               solver used expected it to be.

             * **-26**

               The requested factorization solver is unavailable.

             * **-29**

               A requested option is unavailable.

             * **-45**

               The requested preconditioner is unavailable.

             * **-80**

               An error occurred when calling ``HSL MI28``. 
               See mi28 info%stat for more details.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          analyse_status : int
             status return from factorization.
          factorize_status : int
             status return from factorization.
          solve_status : int
             status return from solution phase.
          factorization_integer : long
             number of integer words to hold factors.
          factorization_real : long
             number of real words to hold factors.
          preconditioner : int
             code for the actual preconditioner used (see
             control.preconditioner).
          semi_bandwidth : int
             the actual semi-bandwidth.
          reordered_semi_bandwidth : int
             the semi-bandwidth following reordering (if any).
          out_of_range : int
             number of indices out-of-range.
          duplicates : int
             number of duplicates.
          upper : int
             number of entries from the strict upper triangle.
          missing_diagonals : int
             number of missing diagonal entries for an
             allegedly-definite matrix.
          semi_bandwidth_used : int
             the semi-bandwidth used.
          neg1 : int
             number of 1 by 1 pivots in the factorization.
          neg2 : int
             number of 2 by 2 pivots in the factorization.
          perturbed : bool
             has the preconditioner been perturbed during the
             fctorization?.
          fill_in_ratio : float
             ratio of fill in to original nonzeros.
          norm_residual : float
             the norm of the solution residual.
          mc61_info : int
             the integer and real output arrays from ``MC61``.
          mc61_rinfo : float
             see mc61_info.
          time : dict
             dictionary containing timing information:
               total : float
                  total time.
               assemble : float
                  time to assemble the preconditioner prior to factorization.
               analyse : float
                  time for the analysis phase.
               factorize : float
                  time for the factorization phase.
               solve : float
                  time for the linear solution phase.
               update : float
                  time to update the factorization.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time to assemble the preconditioner prior to
                  factorization.
               clock_analyse : float
                  clock time for the analysis phase.
               clock_factorize : float
                  clock time for the factorization phase.
               clock_solve : float
                  clock time for the linear solution phase.
               clock_update : float
                  clock time to update the factorization.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          mi28_info : dict
             info parameters for HSL_MI28 (see ``mi28.information``).

   .. function:: psls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/psls/Python/test_psls.py
   :code: python

This example code is available in $GALAHAD/src/psls/Python/test_psls.py .
