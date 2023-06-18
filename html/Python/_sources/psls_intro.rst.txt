purpose
-------

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

  A. R. Conn, N. I. M. Gould and Ph. L. Toint,
  LANCELOT. A fortran package for large-scale nonlinear optimization
  (release A). Springer Verlag Series in Computational Mathematics 17,
  Berlin (1992), Section 3.3.10,

along with the more modern versions implements in ``ICFS`` due to

  C.-J. Lin and J. J. More',
  ``Incomplete Cholesky factorizations with limited memory''.
  *SIAM Journal on Scientific Computing* **21** (1999) 21-45,

and in ``HSL_MI28`` described by

  J. A. Scott and M. Tuma, 
  ``HSL_MI28: an efficient and robust
  limited-memory incomplete Cholesky factorization code''.
  *ACM Transactions on Mathematical Software* **40(4)** (2014), Article 24.

The factorization methods used by the package ``sls`` in conjunction
with some preconditioners are described in the documentation to that
package. The key features of the external solvers supported by ``sls`` 
are given in the following table.

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
for more details and references.
Dummy instances are provided for solvers that are unavailable.
Also note that additional flexibility may be obtained by calling the
solvers directly rather that via this package.

Orderings to reduce the bandwidth, as implemented in HSL's ``MC61``, are due to

  J. K. Reid and J. A. Scott,
  ``Ordering symmetric sparse matrices for small profile and wavefront''.
  *International Journal for Numerical Methods in Engineering*
  **45**  (1999) 1737-1755.

If a subset of the rows and columns are specified, the remaining rows/columns
are removed before processing. Any subsequent removal of rows and columns
is achieved using the Schur-complement updating package ``scu``
unless a complete re-factorization is likely more efficient.
