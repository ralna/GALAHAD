ULS
===

.. module:: galahad.uls

.. include:: uls_intro.rst

.. include:: uls_storage.rst

functions
---------

   .. function:: uls.initialize(solver)

      Set default option values and initialize private data

      **Parameters:**

      solver : str
        the name of the solver required to solve $Ax=b$. 
        It should be one of 'gls', 'ma28', 'ma48' or 'getr';
        lower or upper case variants are allowed.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          warning : int
             unit for warning messages.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output.

             * **1**

               gives a summary of the process.

             * **>=2**

               gives increasingly verbose (debugging) output.

          print_level_solver : int
             controls level of diagnostic output from external solver.
          initial_fill_in_factor : int
             prediction of factor by which the fill-in will exceed the
             initial number of nonzeros in $A$.
          min_real_factor_size : int
             initial size for real array for the factors and other data.
          min_integer_factor_size : int
             initial size for integer array for the factors and other
             data.
          max_factor_size : long
             maximum size for real array for the factors and other data.
          blas_block_size_factorize : int
             level 3 blocking in factorize.
          blas_block_size_solve : int
             level 2 and 3 blocking in solve.
          pivot_control : int
             pivot control. Possible values are

             * **1**

               Threshold Partial Pivoting is desired.

             * **2**

               Threshold Rook Pivoting is desired.

             * **3**

               Threshold Complete Pivoting is desired.

             * **4**

               Threshold Symmetric Pivoting is desired.

             * **5**

               Threshold Diagonal Pivoting is desired.

          pivot_search_limit : int
             number of rows/columns pivot selection restricted to 
             (0 = no restriction).
          minimum_size_for_btf : int
             the minimum permitted size of blocks within the
             block-triangular form.
          max_iterative_refinements : int
             maximum number of iterative refinements allowed.
          stop_if_singular : bool
             stop if the matrix is found to be structurally singular.
          array_increase_factor : float
             factor by which arrays sizes are to be increased if they
             are too small.
          switch_to_full_code_density : float
             switch to full code when the density exceeds this factor.
          array_decrease_factor : float
             if previously allocated internal workspace arrays are
             greater than array_decrease_factor times the currently
             required sizes, they are reset to current requirements.
          relative_pivot_tolerance : float
             pivot threshold.
          absolute_pivot_tolerance : float
             any pivot small than this is considered zero.
          zero_tolerance : float
             any entry smaller than this in modulus is reset to zero.
          acceptable_residual_relative : float
             refinement will cease as soon as the residual $\|Ax-b\|$
             falls below max( acceptable_residual_relative * $\|b\|$,
             acceptable_residual_absolute ).
          acceptable_residual_absolute : float
             see acceptable_residual_relative.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: uls.factorize_matrix(m, n, A_type, A_ne, A_row, A_col, A_ptr, A_val, options=None)

      Import problem data into internal storage, compute a sparsity-based 
      reorderings prior to factorization, and then factorize the matrix.

      **Parameters:**

      m : int
          holds the number of rows of $A$.
      n : int
          holds the number of columns of $A$.
      A_type : string
          specifies the symmetric storage scheme used for the matrix $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in the matrix
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      A_row : ndarray(A_ne)
          holds the row indices of the matrix $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of the matrix
          $A$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      A_ptr : ndarray(n+1)
          holds the starting position of each row of the matrix $A$, 
          as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the matrix
          $A$ in the same order as specified for A_row, A_col and A_ptr above,
      options : dict, optional
          dictionary of control options (see ``uls.initialize``).

   .. function:: uls.solve_system(m, n, b, trans)

      Given the factors of $A$, solve the system of linear equations $Ax=b$.

      **Parameters:**

      m : int
          holds the number of rows of $A$.
      n : int
          holds the number of columns of $A$.
      b : ndarray(n) if ``trans`` in False or ndarray(m) if ``trans`` in True.
          holds the values of the right-hand side vector $b$
      trans : bool
          should be True if the solution to $A^T x = b$ is required or
          False if the solution to $A x = b$ is desired.

      **Returns:**

      x : ndarray(m) if ``trans`` in False or ndarray(n) if ``trans`` in True.
          holds the values of the solution $x$ after a successful call.

   .. function:: [optional] uls.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             reported return status. Possible values are

             * **0**

               success

             * **-1**

               allocation error

             * **-2**

               deallocation error

             * **-3**

               matrix data faulty (m < 1, n < 1, ne < 0)

             * **-26**

               unknown solver

             * **-29**

               unavailable option

             * **-31**

               input order is not a permutation or is faulty in
               some other way

             * **-32**

               error with integer workspace

             * **-33**

               error with real workspace

             * **-50**

               solver-specific error; see the solver's info
               parameter.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          more_info : int
             further information on failure.
          out_of_range : long
             number of indices out-of-range.
          duplicates : long
             number of duplicates.
          entries_dropped : long
             number of entries dropped during the factorization.
          workspace_factors : long
             predicted or actual number of reals and integers to hold
             factors.
          compresses : int
             number of compresses of data required.
          entries_in_factors : long
             number of entries in factors.
          rank : int
             estimated rank of the matrix.
          structural_rank : int
             structural rank of the matrix.
          pivot_control : int
             pivot control. Possible values are

             * **1**

               Threshold Partial Pivoting has been used.

             * **2**

               Threshold Rook Pivoting has been used.

             * **3**

               Threshold Complete Pivoting has been desired.

             * **4**

               Threshold Symmetric Pivoting has been desired.

             * **5**

               Threshold Diagonal Pivoting has been desired.

          iterative_refinements : int
             number of iterative refinements performed.
          alternative : bool
             has an "alternative" $y: A^T y = 0$ and $y^T b > 0$ been found
             when trying to solve $A x = b$ ?.
          gls_ainfo : dict
             the output arrays from GLS.
          gls_finfo : dict
             see gls_ainfo.
          gls_sinfo : dict
             see gls_ainfo.
          ma48_ainfo : dict
             the output arrays from MA48.
          ma48_finfo : dict
             see ma48_ainfo.
          ma48_sinfo : dict
             see ma48_ainfo.
          lapack_error : int
             the LAPACK error return code.

   .. function:: uls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/uls/Python/test_uls.py
   :code: python

This example code is available in $GALAHAD/src/uls/Python/test_uls.py .
