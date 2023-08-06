purpose
-------

The ``uls`` package 
**solves dense or sparse unsymmetric systems of linear equations**
using variants of Gaussian elimination.
Given a sparse matrix $A = \{ a_{ij} \}_{m \times n}$, and an
$n$-vector $b$, this function solves the systems $A x = b$ or $A^T x = b$.
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

.. _details-uls__solvers:

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
