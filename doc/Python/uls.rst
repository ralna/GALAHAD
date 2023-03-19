ULS
===

.. module:: galahad.uls

The ``uls`` package 
**solves dense or sparse unsymmetric systems of linear equations**
using variants of Gaussian elimination.
Given a sparse symmetric matrix $A = \{ a_{ij} \}_{m \times n}$, and an
$n$-vector $b$, this function solves the systems $A x = b$ or $A^T x = b$
Both square ($m=n$) and rectangular ($m\neq n$)  matrices are handled; 
one of an infinite class of  solutions for consistent systems will be returned
whenever $A$ is not of full rank.

The method provides a common interface to a variety of well-known
solvers from HSL and elsewhere. Currently supported solvers include
``MA28/GLS`` and ``HSL_MA48`` from {HSL},
as well as ``GETR`` from LAPACK.
Note that, with the exception of he Netlib reference LAPACK code,
**the solvers themselves do not form part of this package and
must be obtained/linked to separately.**
Dummy instances are provided for solvers that are unavailable.
Also note that additional flexibility may be obtained by calling the
solvers directly rather that via this package.

terminology
-----------

The solvers used each produce an $L U$ factorization of
$A$, where $L$ and U are permuted
lower and upper triangular matrices (respectively).
It is convenient to write this factorization in the form
$$A = P_R L U P_C,$$
where $P_R$ and $P_C$ are row and column permutation matrices, respectively.

supported solvers
-----------------

The key features of the external solvers supported by ``uls`` are
given in the following table:

.. list-table:: External solver characteristics
   :widths: 50 50 40 80
   :header-rows: 1

   * - solver 
     - factorization 
     - out-of-core 
     - parallelised
   * - ``GLS/MA28`` 
     - sparse
     - no 
     - no
   * - ``HSL_MA48`` 
     - sparse
     - no 
     - no
   * - ``GETR`` 
     - dense
     - no 
     - with parallel LAPACK

method
------

Variants of sparse Gaussian elimination are used.
See Section 4 of $GALAHAD/doc/uls.pdf for a brief description of the
method employed and other details.

The solver ``GLS`` is available as part of GALAHAD and relies on
the HSL Archive package ``MA33``. To obtain HSL Archive packages, see

- http://hsl.rl.ac.uk/archive/ .

The solver ``HSL_MA48`` is part of HSL 2011.
To obtain HSL 2011 packages, see

- http://hsl.rl.ac.uk .

The solver ``GETR`` is available as ``S/DGETRF/S``
as part of LAPACK. Reference versions
are provided by GALAHAD, but for good performance
machined-tuned versions should be used.

The methods used are described in the user-documentation for

HSL 2011, A collection of Fortran codes for large-scale scientific computation (2011). 

- http://www.hsl.rl.ac.uk .


matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ may be presented
and stored in a variety of convenient input formats. 

*Dense* storage format:
The matrix $A$ is stored as a compact dense matrix by rows, that is,
the values of the entries of each row in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $n \ast i + j$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense' should be specified.

*Dense by columns* storage format:
The matrix $A$ is stored as a compact dense matrix by columns, that is,
the values of the entries of each column in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $m \ast j + i$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1
The string A_type = 'dense_by_columns' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as the $l$-th 
components of the integer arrays A_row and A_col and real array A_val, 
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
The string A_type = 'coordinate'should be specified.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries plus one.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.
For sparse matrices, this scheme almost always requires less storage than
its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Sparse column-wise* storage format:
Once again only the nonzero entries are stored, but this time
they are ordered so that those in column j appear directly before those
in column j+1. For the j-th column of $A$ the j-th component of the
integer array A_ptr holds the position of the first entry in this column,
while A_ptr(n) holds the total number of entries plus one.
The row indices i, $0 \leq i \leq m-1$, and values $A_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = A_ptr(j), $\ldots$, A_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array A_row, and real array A_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.
The string A_type = 'sparse_by_columns' should be specified.

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
          as well as the total number of entries plus one,
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
             error ocurred.
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
