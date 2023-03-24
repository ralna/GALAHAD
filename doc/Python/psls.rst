PSLS
====

.. module:: galahad.psls

Given a sparse symmetric matrix $A = \{ a_{ij} \}_{n \times n}$, 
the ``psls`` package **builds a suitable symmetric, positive definite---or 
diagonally dominant---preconditioner $P$ of $A$** or a symmetric 
sub-matrix thereof. The matrix $A$ need not be definite. Facilities
are provided to apply the preconditioner to a given vector, and to
remove rows and columns (symmetrically) from the initial preconditioner
without a full re-factorization.


method
------

The basic preconditioners are described in detail in 

A. R. Conn, N. I. M. Gould and Ph. L. Toint (1992).
LANCELOT. A fortran package for large-scale nonlinear optimization
(release A). Springer Verlag Series in Computational Mathematics 17,
Berlin, Section 3.3.10,

along with the more modern versions implements in ``ICFS`` due to

C.-J. Lin and J. J. More' (1999).
Incomplete Cholesky factorizations with limited memory.
SIAM Journal on Scientific Computing **21** 21-45,

and in ``HSL_MI28`` described by

J. A. Scott and M. Tuma (2013). HSL_MI28: an efficient and robust
limited-memory incomplete Cholesky factorization code.
ACM Transactions on Mathematical Software **40(4)** (2014), Article 24.

The factorization methods used by the GALAHAD package ``sls`` in conjunction
with some preconditioners are described in the documentation to that
package. The key features of the external solvers supported by ``sls`` are
given in the following table.

.. list-table:: External solver characteristics
   :widths: 50 50 40 40 80
   :header-rows: 1

   * - solver 
     - factorization 
     - indefinite $A$ 
     - out-of-core 
     - parallelised
   * - ``SILS/MA27`` 
     - multifrontal 
     - yes 
     - no 
     - no
   * - ``HSL_MA57`` 
     - multifrontal 
     - yes 
     - no 
     - no
   * - ``HSL_MA77`` 
     - multifrontal 
     - yes 
     - yes 
     - OpenMP core
   * - ``HSL_MA86`` 
     - left-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``HSL_MA87`` 
     - left-looking 
     - no 
     - no 
     - OpenMP fully
   * - ``HSL_MA97`` 
     - multifrontal 
     - yes 
     - no 
     - OpenMP core
   * - ``SSIDS`` 
     - multifrontal 
     - yes 
     - no 
     - CUDA core
   * - ``MUMPS`` 
     - multifrontal 
     - yes 
     - optionally 
     - MPI
   * - ``PARDISO`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``MKL_PARDISO`` 
     - left-right-looking 
     - yes 
     - optionally 
     - OpenMP fully
   * - ``PaStix`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``WSMP`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``POTR`` 
     - dense 
     - no 
     - no 
     - with parallel LAPACK
   * - ``SYTR`` 
     - dense 
     - yes 
     - no 
     - with parallel LAPACK
   * - ``PBTR`` 
     - dense band 
     - no 
     - no 
     - with parallel LAPACK

Note that, with the exception of ``SSIDS`` and the Netlib
reference LAPACK codes,
**the solvers themselves do not form part of this package and
must be obtained/linked to separately.** See the documentation for ``sls``
for more details.
Dummy instances are provided for solvers that are unavailable.
Also note that additional flexibility may be obtained by calling the
solvers directly rather that via this package.

Orderings to reduce the bandwidth, as implemented in HSL's ``MC61``, are due to

J. K. Reid and J. A. Scott (1999)
Ordering symmetric sparse matrices for small profile and wavefront
International Journal for Numerical Methods in Engineering
**45** 1737-1755.

If a subset of the rows and columns are specified, the remaining rows/columns
are removed before processing. Any subsequent removal of rows and columns
is achieved using the GALAHAD Schur-complement updating package ``scu``
unless a complete re-factorization is likely more efficient.

matrix storage
--------------

The **symmetric** $n$ by $n$ matrix $A$ may 
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $A$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $A$ is
symmetric, only the lower triangular part (that is the part
$A_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array A_val
will hold the value $A_{ij}$ (and, by symmetry, $A_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string A_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays A_row and A_col and real array A_val,
respectively, while the number of nonzeros is recorded as
A_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string A_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(n) holds the total number of entries plus one.
The column indices j, $0 \leq j \leq i$, and values
$A_{ij}$ of the  entries in the i-th row are stored in components
l = A_ptr(i), ..., A_ptr(i+1)-1 of the
integer array A_col, and real array A_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $A$ is diagonal (i.e., $A_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$A_{ii}$, $0 \leq i \leq n-1$ need be stored, 
and the first n components of the array A_val may be used for the purpose.
The string A_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $A$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of A_val.
The string A_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $A$ is the identity matrix, no values need be stored.
The string A_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $A$ is the zero matrix, but now
the string A_type = 'zero' or 'none' should be specified.


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
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
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

   .. function:: psls.import(n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          part of $A$, as well as the total number of entries plus one,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``psls.initialize``).

   .. function:: psls.form_preconditioner(a_ne, A_val)

      Form and factorize the preconditioner $P$ from the matrix $A$.

      **Parameters:**

      a_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``psls.load``.

   .. function:: psls.form_subset_preconditioner(a_ne, A_val,n_sub,sub)

      Form and factorize the preconditioner of a symmetric subset of the
      rows and columns of $A$.

      **Parameters:**

      a_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(a_ne)
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
             error ocurred.
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
