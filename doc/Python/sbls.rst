SBLS
====

.. module:: galahad.sbls

Given a **block, symmetric matrix**
$$K_{H} = \left(\begin{matrix}H & A^T \\ A  & - C\end{matrix}\right),$$
the ``sbls`` package constructs a variety of **preconditioners** of the form
$$K_{G} = \left(\begin{matrix}G & A^T \\ A  & - C\end{matrix}\right).$$
Here, the leading-block matrix $G$ is a suitably-chosen
approximation to $H$; it may either be prescribed **explicitly**, in
which case a symmetric indefinite factorization of $K_{G}$
will be formed using the \galahad\ package {\tt SLS},
or **implicitly**, by requiring certain sub-blocks of $G$
be zero. In the latter case, a factorization of $K_{G}$ will be
obtained implicitly (and more efficiently) without recourse to ``sls``.
In particular, for implicit preconditioners, a reordering
$$K_{G} = P
\left(\begin{matrix}G_{11}^{} & G_{21}^T & A_1^T \\ G_{21}^{} & G_{22}^{} & A_2^T \\ A_{1}^{} & A_{2}^{} & - C\end{matrix}\right) P^T$$
involving a suitable permutation $P$ will be found, for some
invertible sub-block ("basis") $A_1$ of the columns of $A$;
the selection and factorization of $A_1$ uses the package ``uls``.
Once the preconditioner has been constructed,
solutions to the preconditioning system
$$\left(\begin{matrix}G & A^T \\ A  & - C\end{matrix}\right) \left(\begin{matrix}x \\ y\end{matrix}\right)= \left(\begin{matrix}a \\ b\end{matrix}\right)$$
may be obtained by the package. Full advantage is taken of any zero 
coefficients in the matrices $H$, $A$ and $C$.

See Section 4 of $GALAHAD/doc/sbls.pdf for a brief description of the
method employed and other details.

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

The **symmetric** $n$ by $n$ matrix $H$, as well as the $m$ by $m$ matrix $C$,
may also be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal). We focus on
$H$, but everything we say applied equally to $C$.

*Dense* storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string H_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $H$,
its row index i, column index j and value $H_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string H_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries plus one.
The column indices j, $0 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string H_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $H$ is diagonal (i.e., $H_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$H_{ii}$, $0 \leq i \leq n-1$ need be stored, 
and the first n components of the array H_val may be used for the purpose.
The string H_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $H$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of H_val.
The string H_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $H$ is the identity matrix, no values need be stored.
The string H_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $H$ is the zero matrix, but now
the string H_type = 'zero' or 'none' should be specified.


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
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
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
          part of $H$, as well as the total number of entries plus one,
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
          total number of entries plus one, in the sparse row-wise storage 
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
          part of $C$, as well as the total number of entries plus one,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``sbls.initialize``).

   .. function:: sbls.factorize_matrix(n, m, h_ne, H_val, a_ne, A_val, c_ne, C_val,D)

      Form and factorize the block matrix
      $$K_{G} = \left(\begin{matrix}G & A^T \\ A  & - C\end{matrix}\right)$$
      for some appropriate matrix $G$.

      **Parameters:**

      n : int
          holds the dimension of $H$ 
          (equivalently, the number of columns of $A$).
      m : int
          holds the dimension of $C$
          (equivalently, the number of rows of $A$).
      h_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $H$.
      H_val : ndarray(h_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $H$ in the same order as specified in the sparsity pattern in 
          ``sbls.load``.
      a_ne : int
          holds the number of entries in the matrix $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``sbls.load``.
      c_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $C$.
      C_val : ndarray(c_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $C$ in the same order as specified in the sparsity pattern in 
          ``sbls.load``.
      D : ndarray(n)
          holds the values of the diagonals of the matrix $D$ that is required 
          if options[`preconditioner`]=5 has been specified. Otherwise it
          shuld be set to None.

   .. function:: sbls.solve_system(n, m, rhs)

      Solve the block linear system
      $$\left(\begin{matrix}G & A^T \\ A  & - C\end{matrix}\right) \left(\begin{matrix}x \\ y\end{matrix}\right)= \left(\begin{matrix}a \\ b\end{matrix}\right)$$


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
             error ocurred.
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
