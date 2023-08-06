purpose
-------

The ``sls`` package 
**solves dense or sparse symmetric systems of linear equations**
using variants of Gaussian elimination.
Given a sparse symmetric matrix $A = \{ a_{ij} \}_{n \times n}$, and an
$n$-vector $b$ or a matrix $B = \{ b_{ij} \}_{n \times r}$, this
function solves the system $A x = b$ or the system $A X = B$ . 
The matrix $A$ need not be definite.

The method provides a common interface to a variety of well-known
solvers from HSL and elsewhere. Currently supported solvers include
``MA27/SILS``, ``HSL_MA57``, ``HSL_MA77`` , ``HSL_MA86``,
``HSL_MA87`` and ``HSL_MA97`` from {HSL},
``SSIDS`` from {SPRAL},
``MUMPS`` from Mumps Technologies,
``PARDISO`` both from the Pardiso Project and Intel's MKL,
``PaStiX`` from Inria, and
``WSMP`` from the IBM alpha Works, 
as well as ``POTR``, ``SYTR`` and ``SBTR`` from LAPACK.
Note that, with the exception of ``SSIDS`` and the Netlib
reference LAPACK codes,
**the solvers themselves do not form part of this package and
must be obtained/linked to separately.**
Dummy instances are provided for solvers that are unavailable.
Also note that additional flexibility may be obtained by calling the
solvers directly rather that via this package.

terminology
-----------

The solvers used each produce an $L D L^T$ factorization of
$A$ or a perturbation thereof, where $L$ is a permuted
lower triangular matrix and $D$ is a block diagonal matrix with
blocks of order 1 and 2. It is convenient to write this factorization in
the form
$$A + E = P L D L^T P^T,$$
where $P$ is a permutation matrix and $E$ is any diagonal
perturbation introduced.

.. _details-sls__solvers:

supported solvers
-----------------

The key features of the external solvers supported by ``sls`` are
given in the following table:

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

method
------

Variants of sparse Gaussian elimination are used.
See Section 4 of $GALAHAD/doc/sls.pdf for a brief description of the
method employed and other details.

The solver ``SILS`` is available as part of GALAHAD and relies on
the HSL Archive package ``MA27``. To obtain HSL Archive packages, see

- http://hsl.rl.ac.uk/archive/ .

The solvers
``HSL_MA57``,
``HSL_MA77``,
``HSL_MA86``,
``HSL_MA87``
and
``HSL_MA97``, the ordering packages
``MC61`` and ``HSL_MC68``, and the scaling packages
``HSL_MC64`` and ``MC77``
are all part of HSL 2011.
To obtain HSL 2011 packages, see

- http://hsl.rl.ac.uk .

The solver ``SSIDS`` is from the SPRAL sparse-matrix collection,
and is available as part of GALAHAD.

The solver ``MUMPS`` is available from Mumps Technologies in France, and 
version 5.5.1 or above is sufficient.
To obtain ``MUMPS``, see

- https://mumps-solver.org .

The solver ``PARDISO`` is available from the Pardiso Project;
version 4.0.0 or above is required.
To obtain ``PARDISO``, see

- http://www.pardiso-project.org/ .

The solver ``MKL PARDISO`` is available as part of Intel's oneAPI Math Kernel
Library (oneMKL).
To obtain this version of ``PARDISO``, see

- https://software.intel.com/content/www/us/en/develop/tools/oneapi.html .

The solver ``PaStix`` is available from Inria in France, and 
version 6.2 or above is sufficient.
To obtain ``PaStiX``, see

- https://solverstack.gitlabpages.inria.fr/pastix .

The solver ``WSMP`` is available from the IBM alpha Works;
version 10.9 or above is required.
To obtain ``WSMP``, see

- http://www.alphaworks.ibm.com/tech/wsmp .

The solvers ``POTR``, ``SYTR`` and ``PBTR``,
are available as
``S/DPOTRF/S``,
``S/DSYTRF/S`` and ``S/DPBTRF/S``
as part of LAPACK. Reference versions
are provided by GALAHAD, but for good performance
machined-tuned versions should be used.

Explicit sparsity re-orderings are obtained by calling the HSL package
``HSL_MC68``.
Both this, ``HSL_MA57`` and ``PARDISO`` rely optionally
on the ordering package ``MeTiS`` (version 4) from the Karypis Lab.
To obtain ``METIS``, see

- http://glaros.dtc.umn.edu/gkhome/views/metis/ .

Bandwidth, Profile and wavefront reduction is supported by
calling HSL's ``MC61``.

The methods used are described in the user-documentation for

HSL 2011, A collection of Fortran codes for large-scale scientific computation (2011). 

- http://www.hsl.rl.ac.uk

and papers

E. Agullo, P. R. Amestoy, A. Buttari, J.-Y. L'Excellent, A. Guermouche 
and F.-H. Rouet,
"Robust memory-aware mappings for parallel multifrontal factorizations".
SIAM Journal on Scientific Computing, \b 38(3) (2016), C256--C279,

P. R. Amestoy, I. S. Duff, J. Koster and J.-Y. L'Excellent.
"A fully asynchronous multifrontal solver using distributed 
dynamic scheduling".
SIAM Journal on Matrix Analysis and Applications \b 23(1) (2001) 15-41,

A. Gupta,
"WSMP: Watson Sparse Matrix Package Part I - direct
solution of symmetric sparse systems".
IBM Research Report RC 21886, IBM T. J. Watson Research Center,
NY 10598, USA (2010),

P. Henon, P. Ramet and J. Roman,
"PaStiX: A High-Performance Parallel Direct Solver for Sparse Symmetric 
Definite Systems".
Parallel Computing, \b 28(2) (2002) 301--321,

J.D. Hogg, E. Ovtchinnikov and J.A. Scott. 
"A sparse symmetric indefinite direct solver for GPU architectures".
ACM Transactions on Mathematical Software \b 42(1) (2014), Article 1,

O. Schenk and K. Gartner,
"Solving Unsymmetric Sparse Systems of Linear Equations with PARDISO".
Journal of Future Generation Computer Systems \b, 20(3) (2004) 475--487,
and

O. Schenk and K. Gartner,
"On fast factorization pivoting methods for symmetric indefinite systems".
Electronic Transactions on Numerical Analysis \b 23 (2006) 158--179.
