//* \file galahad_sls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 27th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package sls

  \section sls_intro Introduction

  \subsection sls_purpose Purpose

  This package
  <b> solves dense or sparse symmetric systems of linear equations</b>
  using variants of Gaussian elimination.  Given a sparse symmetric
  \f$n \times n\f$ matrix \f$A\f$, and an \f$n\f$-vector \f$b\f$, this
  subroutine solves the system \f$A x = b\f$.  The matrix \f$A\f$ need not
  be definite.

  The package provides a common interface to a variety of well-known
  solvers from HSL and elsewhere. Currently supported solvers include
  \c MA27/SILS, \c HSL\_MA57, \c HSL\_MA77, \c HSL\_MA86,
  \c HSL\_MA87 and \c HSL\_MA97 from HSL,
  \c SSIDS from SPRAL,
  \c MUMPS from Mumps Technologies,
  \c PARDISO both from the Pardiso Project and Intel's MKL,
  \c PaStix from Inria
  and \c WSMP from the IBM alpha Works, as
  well as \c POTR, \c SYTR and \c SBTR from LAPACK.
  Note that
  <b> the solvers themselves do not form part of this package and
  must be obtained separately.</b>
  Dummy instances are provided for solvers that are unavailable.
  Also note that additional flexibility may be obtained by calling the
  solvers directly rather that via this package.

  \subsection sls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sls_date Originally released

  August 2009, C interface December 2021.

  \subsection sls_terminology Terminology

  The solvers used each produce an \f$L D L^T\f$ factorization of
  \f$A\f$ or a perturbation thereof, where \f$L\f$ is a permuted
  lower triangular matrix and \f$D\f$ is a block diagonal matrix with
  blocks of order 1 and 2. It is convenient to write this factorization in
  the form
  \f[A + E = P L D L^T P^T,\f] where
  \f$P\f$ is a permutation matrix and \f$E\f$ is any diagonal
  perturbation introduced.

  \subsection sls_solvers Supported external solvers

  The key features of the external solvers supported by sls are
  given in the following table.

\manonly
(ignore next paragraph - doxygen bug!)
\endmanonly

<table>
<caption>External solver characteristics</caption>
<tr><th> solver <th> factorization <th> indefinite \f$A\f$
    <th> out-of-core <th> parallelised
<tr><td> \c SILS/MA27 <td> multifrontal <td> yes <td> no <td> no
<tr><td> \c HSL_MA57 <td> multifrontal <td> yes <td> no <td> no
<tr><td> \c HSL_MA77 <td> multifrontal <td> yes <td> yes <td> OpenMP core
<tr><td> \c HSL_MA86 <td> left-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c HSL_MA87 <td> left-looking <td> no <td> no <td> OpenMP fully
<tr><td> \c HSL_MA97 <td> multifrontal <td> yes <td> no <td> OpenMP core
<tr><td> \c SSIDS <td> multifrontal <td> yes <td> no <td> CUDA core
<tr><td> \c MUMPS <td> multifrontal <td> yes <td> optionally <td> MPI
<tr><td> \c PARDISO <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c MKL_PARDISO <td> left-right-looking <td> yes <td> optionally
     <td> OpenMP fully
<tr><td> \c PaStix <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c WSMP <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c POTR <td> dense <td> no <td> no <td> with parallel LAPACK
<tr><td> \c SYTR <td> dense <td> yes <td> no <td> with parallel LAPACK
<tr><td> \c PBTR <td> dense band <td> no <td> no <td> with parallel LAPACK
</table>

\manonly
External solver characteristics (ooc = out-of-core factorization)

 solver     factorization indefinite A  ooc    parallelised
 SILS/MA27   multifrontal          yes   no    no
 HSL_MA57    multifrontal          yes   no    no
 HSL_MA77    multifrontal          yes  yes    OpenMP core
 HSL_MA86    left-looking          yes   no    OpenMP fully
 HSL_MA87    left-looking           no   no    OpenMP fully
 HSL_MA97    multifrontal          yes   no    OpenMP core
 SSIDS       multifrontal          yes   no    CUDA core
 MUMPS       multifrontal          yes  optionally  MPI
 PARDISO     left-right-looking    yes   no    OpenMP fully
 MKL_PARDISO left-right-looking    yes  optionally  OpenMP fully
 PaStix      left-right-looking    yes   no    OpenMP fully
 WSMP        left-right-looking    yes   no    OpenMP fully
 POTR        dense                  no   no    with parallel LAPACK
 SYTR        dense                 yes   no    with parallel LAPACK
 PBTR        dense band             no   no    with parallel LAPACK
\endmanonly

  \subsection sls_method Method

Variants of sparse Gaussian elimination are used.

The solver \c SILS is available as part of GALAHAD and relies on
the HSL Archive package \c MA27. To obtain HSL Archive packages, see

http://hsl.rl.ac.uk/archive/ .

The solvers
\c HSL\_MA57,
\c HSL\_MA77,
\c HSL\_MA86,
\c HSL\_MA87
and
\c HSL\_MA97, the ordering packages
\c MC61 and \c HSL\_MC68, and the scaling packages
\c HSL\_MC64 and \c MC77
are all part of HSL 2011.
To obtain HSL 2011 packages, see

http://hsl.rl.ac.uk

The solver \c SSIDS is from the SPRAL sparse-matrix collection,
and is available as part of GALAHAD.

The solver \c MUMPS is available from Mumps Technologies in France, and
version 5.5.1 or above is sufficient.
To obtain \c MUMPS, see

https://mumps-solver.org .

The solver \c PARDISO is available from the Pardiso Project;
version 4.0.0 or above is required.
To obtain \c PARDISO, see

http://www.pardiso-project.org/ .

The solver \c MKL PARDISO is available as part of Intel's oneAPI Math Kernel
Library (oneMKL).
To obtain this version of \c PARDISO, see

https://software.intel.com/content/www/us/en/develop/tools/oneapi.html .

The solver \c PaStix is available from Inria in France, and
version 6.2 or above is sufficient.
To obtain \c PaStiX, see

https://solverstack.gitlabpages.inria.fr/pastix .

The solver \c WSMP is available from the IBM alpha Works;
version 10.9 or above is required.
To obtain \c WSMP, see

http://www.alphaworks.ibm.com/tech/wsmp .

The solvers \c POTR, \c SYTR and \c PBTR,
are available as
\c S/DPOTRF/S,
\c S/DSYTRF/S and \c S/DPBTRF/S
as part of LAPACK. Reference versions
are provided by GALAHAD, but for good performance
machined-tuned versions should be used.

Explicit sparsity re-orderings are obtained by calling the HSL package
\c HSL\_MC68.
Both this, \c HSL\_MA57 and \c PARDISO rely optionally
on the ordering package \c MeTiS (version 4) from the Karypis Lab.
To obtain \c METIS, see

http://glaros.dtc.umn.edu/gkhome/views/metis/ .

Bandwidth, Profile and wavefront reduction is supported by
calling HSL's \c MC61.

  \subsection sls_references Reference

The methods used are described in the user-documentation for

HSL 2011, A collection of Fortran codes for large-scale scientific
 computation (2011). http://www.hsl.rl.ac.uk

and papers

E. Agullo, P. R. Amestoy, A. Buttari, J.-Y. L'Excellent, A. Guermouche
and F.-H. Rouet,
``Robust memory-aware mappings for parallel multifrontal factorizations''.
SIAM Journal on Scientific Computing, \b 38(3) (2016), C256--C279,

P. R. Amestoy, I. S. Duff, J. Koster and J.-Y. L'Excellent.
``A fully asynchronous multifrontal solver using distributed
dynamic scheduling''.
SIAM Journal on Matrix Analysis and Applications \b 23(1) (2001) 15-41,

A. Gupta,
``WSMP: Watson Sparse Matrix Package Part I - direct
solution of symmetric sparse systems''.
IBM Research Report RC 21886, IBM T. J. Watson Research Center,
NY 10598, USA (2010),

P. Henon, P. Ramet and J. Roman,
``PaStiX: A High-Performance Parallel Direct Solver for Sparse Symmetric
Definite Systems''.
Parallel Computing, \b 28(2) (2002) 301--321,

J.D. Hogg, E. Ovtchinnikov and J.A. Scott.
``A sparse symmetric indefinite direct solver for GPU architectures''.
ACM Transactions on Mathematical Software \b 42(1) (2014), Article 1,

O. Schenk and K. G&auml;rtner,
``Solving Unsymmetric Sparse Systems of Linear Equations with PARDISO''.
Journal of Future Generation Computer Systems \b, 20(3) (2004) 475--487,
and

O. Schenk and K. G&auml;rtner,
``On fast factorization pivoting methods for symmetric indefinite systems''.
Electronic Transactions on Numerical Analysis \b 23 (2006) 158--179.

  \subsection sls_call_order Call order

  To solve a given problem, functions from the sls package must be called
  in the following order:

  - \link sls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link sls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link sls_analyse_matrix \endlink - set up matrix data structures
       and analyse the structure to choose a suitable order for factorization
  - \link sls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link sls_factorize_matrix \endlink - form and factorize the
      matrix \f$A\f$
  - one of
    - \link sls_solve_system \endlink - solve the linear system of
        equations \f$Ax=b\f$
    - \link sls_partial_solve_system \endlink - solve a linear system
        \f$Mx=b\f$ involving one of the matrix factors \f$M\f$ of \f$A\f$
  - \link sls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link sls_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  The symmetric \f$n\f$ by \f$n\f$ coefficient matrix \f$A\f$ may be presented
  and stored in a variety of convenient input formats.  Crucially symmetry
  is exploited  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$A\f$ is symmetric, only the lower triangular part (that is the part
  \f$A_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array val
  will hold the value \f$A_{ij}\f$ (and, by symmetry, \f$A_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays row and
  col and real array val, respectively, while the number of nonzeros
  is recorded as ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A\f$ the i-th component of the
  integer array ptr holds the position of the first entry in this row,
  while ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$A_{ij}\f$ of the  entries in the i-th row are stored in components
  l = ptr(i), \f$\ldots\f$, ptr(i+1)-1 of the
  integer array col, and real array val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SLS_H
#define GALAHAD_SLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sils.h"
#include "hsl_ma57.h"
#include "hsl_ma77.h"
#include "hsl_ma86.h"
#include "hsl_ma87.h"
#include "hsl_ma97.h"
#include "spral_ssids.h"
#include "hsl_mc64.h"
#include "hsl_mc68.h"

/**
 * control derived type as a C struct
 */
struct sls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    ipc_ error;

    /// \brief
    /// unit for warning messages
    ipc_ warning;

    /// \brief
    /// unit for monitor output
    ipc_ out;

    /// \brief
    /// unit for statistical output
    ipc_ statistics;

    /// \brief
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// controls level of diagnostic output from external solver
    ipc_ print_level_solver;

    /// \brief
    /// number of bits used in architecture
    ipc_ bits;

    /// \brief
    /// the target blocksize for kernel factorization
    ipc_ block_size_kernel;

    /// \brief
    /// the target blocksize for parallel elimination
    ipc_ block_size_elimination;

    /// \brief
    /// level 3 blocking in factorize
    ipc_ blas_block_size_factorize;

    /// \brief
    /// level 2 and 3 blocking in solve
    ipc_ blas_block_size_solve;

    /// \brief
    /// a child node is merged with its parent if they both involve fewer than
    /// node_amalgamation eliminations
    ipc_ node_amalgamation;

    /// \brief
    /// initial size of task-pool arrays for parallel elimination
    ipc_ initial_pool_size;

    /// \brief
    /// initial size for real array for the factors and other data
    ipc_ min_real_factor_size;

    /// \brief
    /// initial size for integer array for the factors and other data
    ipc_ min_integer_factor_size;

    /// \brief
    /// maximum size for real array for the factors and other data
    int64_t max_real_factor_size;

    /// \brief
    /// maximum size for integer array for the factors and other data
    int64_t max_integer_factor_size;

    /// \brief
    /// amount of in-core storage to be used for out-of-core factorization
    int64_t max_in_core_store;

    /// \brief
    /// factor by which arrays sizes are to be increased if they are too small
    rpc_ array_increase_factor;

    /// \brief
    /// if previously allocated internal workspace arrays are greater than
    /// array_decrease_factor times the currently required sizes, they are reset
    /// to current requirements
    rpc_ array_decrease_factor;

    /// \brief
    /// pivot control:
    /// \li 1  Numerical pivoting will be performed.
    /// \li 2  No pivoting will be performed and an error exit will
    ///        occur immediately a pivot sign change is detected.
    /// \li 3  No pivoting will be performed and an error exit will
    ///        occur if a zero pivot is detected.
    /// \li 4  No pivoting is performed but pivots are changed to all be
    ///        positive
    ipc_ pivot_control;

    /// \brief
    /// controls ordering (ignored if explicit PERM argument present)
    /// \li <0  chosen by the specified solver with its
    ///        own ordering-selected value -ordering
    /// \li 0  chosen package default (or the AMD ordering if no package
    ///        default)
    /// \li 1  Approximate minimum degree (AMD) with provisions for "dense"
    ///        rows/col
    /// \li 2  Minimum degree
    /// \li 3  Nested disection
    /// \li 4  indefinite ordering to generate a combination of 1x1
    ///        and 2x2 pivots
    /// \li 5  Profile/Wavefront reduction
    /// \li 6  Bandwidth reduction
    /// \li >6  ordering chosen depending on matrix characteristics
    ///        (not yet implemented)
    ipc_ ordering;

    /// \brief
    /// controls threshold for detecting full rows in analyse, registered as
    /// percentage of matrix order. If 100, only fully dense rows detected (defa
    ipc_ full_row_threshold;

    /// \brief
    /// number of rows searched for pivot when using indefinite ordering
    ipc_ row_search_indefinite;

    /// \brief
    /// controls scaling (ignored if explicit SCALE argument present)
    /// \li <0  chosen by the specified solver with its
    ///        own scaling-selected value -scaling
    /// \li 0  No scaling
    /// \li 1  Scaling using HSL's MC64
    /// \li 2  Scaling using HSL's MC77 based on the row one-norm
    /// \li 3  Scaling using HSL's MC77 based on the row infinity-norm
    ipc_ scaling;

    /// \brief
    /// the number of scaling iterations performed (default 10 used if
    /// .scale_maxit < 0)
    ipc_ scale_maxit;

    /// \brief
    /// the scaling iteration stops as soon as the row/column norms are less
    /// than 1+/-.scale_thresh
    rpc_ scale_thresh;

    /// \brief
    /// pivot threshold
    rpc_ relative_pivot_tolerance;

    /// \brief
    /// smallest permitted relative pivot threshold
    rpc_ minimum_pivot_tolerance;

    /// \brief
    /// any pivot small than this is considered zero
    rpc_ absolute_pivot_tolerance;

    /// \brief
    /// any entry smaller than this is considered zero
    rpc_ zero_tolerance;

    /// \brief
    /// any pivot smaller than this is considered zero for positive-definite sol
    rpc_ zero_pivot_tolerance;

    /// \brief
    /// any pivot smaller than this is considered to be negative for p-d solvers
    rpc_ negative_pivot_tolerance;

    /// \brief
    /// used for setting static pivot level
    rpc_ static_pivot_tolerance;

    /// \brief
    /// used for switch to static
    rpc_ static_level_switch;

    /// \brief
    /// used to determine whether a system is consistent when seeking a Fredholm
    /// alternative
    rpc_ consistency_tolerance;

    /// \brief
    /// maximum number of iterative refinements allowed
    ipc_ max_iterative_refinements;

    /// \brief
    /// refinement will cease as soon as the residual ||Ax-b|| falls below
    /// max( acceptable_residual_relative * ||b||, acceptable_residual_absolute
    rpc_ acceptable_residual_relative;
    /// see acceptable_residual_relative
    rpc_ acceptable_residual_absolute;

    /// \brief
    /// set .multiple_rhs to .true. if there is possibility that the solver
    /// will be required to solve systems with more than one right-hand side.
    /// More efficient execution may be possible when  .multiple_rhs = .false.
    bool multiple_rhs;

    /// \brief
    /// if .generate_matrix_file is .true. if a file describing the current
    /// matrix is to be generated
    bool generate_matrix_file;

    /// \brief
    /// specifies the unit number to write the input matrix (in co-ordinate form
    ipc_ matrix_file_device;

    /// \brief
    /// name of generated matrix file containing input problem
    char matrix_file_name[31];

    /// \brief
    /// directory name for out of core factorization
    /// and additional real workspace in the indefinite case, respectively
    char out_of_core_directory[401];

    /// \brief
    /// out of core superfile names for integer and real factor data, real works
    /// and additional real workspace in the indefinite case, respectively
    char out_of_core_integer_factor_file[401];
    /// see out_of_core_integer_factor_file
    char out_of_core_real_factor_file[401];
    /// see out_of_core_integer_factor_file
    char out_of_core_real_work_file[401];
    /// see out_of_core_integer_factor_file
    char out_of_core_indefinite_file[401];
    /// see out_of_core_integer_factor_file
    char out_of_core_restart_file[501];

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];
};

/**
 * time derived type as a C struct
 */
struct sls_time_type {

    /// \brief
    /// the total cpu time spent in the package
    rpc_ total;

    /// \brief
    /// the total cpu time spent in the analysis phase
    rpc_ analyse;

    /// \brief
    /// the total cpu time spent in the factorization phase
    rpc_ factorize;

    /// \brief
    /// the total cpu time spent in the solve phases
    rpc_ solve;

    /// \brief
    /// the total cpu time spent by the external solver in the ordering phase
    rpc_ order_external;

    /// \brief
    /// the total cpu time spent by the external solver in the analysis phase
    rpc_ analyse_external;

    /// \brief
    /// the total cpu time spent by the external solver in the factorization pha
    rpc_ factorize_external;

    /// \brief
    /// the total cpu time spent by the external solver in the solve phases
    rpc_ solve_external;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the total clock time spent in the analysis phase
    rpc_ clock_analyse;

    /// \brief
    /// the total clock time spent in the factorization phase
    rpc_ clock_factorize;

    /// \brief
    /// the total clock time spent in the solve phases
    rpc_ clock_solve;

    /// \brief
    /// the total clock time spent by the external solver in the ordering phase
    rpc_ clock_order_external;

    /// \brief
    /// the total clock time spent by the external solver in the analysis phase
    rpc_ clock_analyse_external;

    /// \brief
    /// the total clock time spent by the external solver in the factorization p
    rpc_ clock_factorize_external;

    /// \brief
    /// the total clock time spent by the external solver in the solve phases
    rpc_ clock_solve_external;
};

/**
 * inform derived type as a C struct
 */
struct sls_inform_type {

    /// \brief
    /// reported return status:
    /// 0  success
    /// -1  allocation error
    /// -2  deallocation error
    /// -3  matrix data faulty (.n < 1, .ne < 0)
    /// -20  alegedly +ve definite matrix is not
    /// -29  unavailable option
    /// -31  input order is not a permutation or is faulty in some other way
    /// -32  > control.max_integer_factor_size integer space required for factor
    /// -33  > control.max_real_factor_size real space required for factors
    /// -40  not possible to alter the diagonals
    /// -41  no access to permutation or pivot sequence used
    /// -42  no access to diagonal perturbations
    /// -43  direct-access file error
    /// -50  solver-specific error; see the solver's info parameter
    /// -101  unknown solver
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// further information on failure
    ipc_ more_info;

    /// \brief
    /// number of entries
    ipc_ entries;

    /// \brief
    /// number of indices out-of-range
    ipc_ out_of_range;

    /// \brief
    /// number of duplicates
    ipc_ duplicates;

    /// \brief
    /// number of entries from the strict upper triangle
    ipc_ upper;

    /// \brief
    /// number of missing diagonal entries for an allegedly-definite matrix
    ipc_ missing_diagonals;

    /// \brief
    /// maximum depth of the assembly tree
    ipc_ max_depth_assembly_tree;

    /// \brief
    /// nodes in the assembly tree (= number of elimination steps)
    ipc_ nodes_assembly_tree;

    /// \brief
    /// desirable or actual size for real array for the factors and other data
    int64_t real_size_desirable;

    /// \brief
    /// desirable or actual size for integer array for the factors and other dat
    int64_t integer_size_desirable;

    /// \brief
    /// necessary size for real array for the factors and other data
    int64_t real_size_necessary;

    /// \brief
    /// necessary size for integer array for the factors and other data
    int64_t integer_size_necessary;

    /// \brief
    /// predicted or actual number of reals to hold factors
    int64_t real_size_factors;

    /// \brief
    /// predicted or actual number of integers to hold factors
    int64_t integer_size_factors;

    /// \brief
    /// number of entries in factors
    int64_t entries_in_factors;

    /// \brief
    /// maximum number of tasks in the factorization task pool
    ipc_ max_task_pool_size;

    /// \brief
    /// forecast or actual size of largest front
    ipc_ max_front_size;

    /// \brief
    /// number of compresses of real data
    ipc_ compresses_real;

    /// \brief
    /// number of compresses of integer data
    ipc_ compresses_integer;

    /// \brief
    /// number of 2x2 pivots
    ipc_ two_by_two_pivots;

    /// \brief
    /// semi-bandwidth of matrix following bandwidth reduction
    ipc_ semi_bandwidth;

    /// \brief
    /// number of delayed pivots (total)
    ipc_ delayed_pivots;

    /// \brief
    /// number of pivot sign changes if no pivoting is used successfully
    ipc_ pivot_sign_changes;

    /// \brief
    /// number of static pivots chosen
    ipc_ static_pivots;

    /// \brief
    /// first pivot modification when static pivoting
    ipc_ first_modified_pivot;

    /// \brief
    /// estimated rank of the matrix
    ipc_ rank;

    /// \brief
    /// number of negative eigenvalues
    ipc_ negative_eigenvalues;

    /// \brief
    /// number of pivots that are considered zero (and ignored)
    ipc_ num_zero;

    /// \brief
    /// number of iterative refinements performed
    ipc_ iterative_refinements;

    /// \brief
    /// anticipated or actual number of floating-point operations in assembly
    int64_t flops_assembly;

    /// \brief
    /// anticipated or actual number of floating-point operations in elimination
    int64_t flops_elimination;

    /// \brief
    /// additional number of floating-point operations for BLAS
    int64_t flops_blas;

    /// \brief
    /// largest diagonal modification when static pivoting or ensuring definiten
    rpc_ largest_modified_pivot;

    /// \brief
    /// minimum scaling factor
    rpc_ minimum_scaling_factor;

    /// \brief
    /// maximum scaling factor
    rpc_ maximum_scaling_factor;

    /// \brief
    /// esimate of the condition number of the matrix (category 1 equations)
    rpc_ condition_number_1;

    /// \brief
    /// estimate of the condition number of the matrix (category 2 equations)
    rpc_ condition_number_2;

    /// \brief
    /// esimate of the backward error (category 1 equations)
    rpc_ backward_error_1;

    /// \brief
    /// esimate of the backward error (category 2 equations)
    rpc_ backward_error_2;

    /// \brief
    /// estimate of forward error
    rpc_ forward_error;

    /// \brief
    /// has an "alternative" y: A y = 0 and yT b > 0 been found when trying to
    /// solve A x = b ?
    bool alternative;

    /// \brief
    /// name of external solver used to factorize and solve
    char solver[21];

    /// \brief
    /// timings (see above)
    struct sls_time_type time;

    /// \brief
    /// the output structure from sils
    struct sils_ainfo_type sils_ainfo;
    /// see sils_ainfo
    struct sils_finfo_type sils_finfo;
    /// see sils_ainfo
    struct sils_sinfo_type sils_sinfo;

    /// \brief
    /// the output structure from ma57
    struct ma57_ainfo ma57_ainfo;
    /// see ma57_ainfo
    struct ma57_finfo ma57_finfo;
    /// see ma57_ainfo
    struct ma57_sinfo ma57_sinfo;

    /// \brief
    /// the output structure from ma77
    struct ma77_info ma77_info;

    /// \brief
    /// the output structure from ma86
    struct ma86_info ma86_info;

    /// \brief
    /// the output structure from ma87
    struct ma87_info ma87_info;

    /// \brief
    /// the output structure from ma97
    struct ma97_info ma97_info;

    /// \brief
    /// the output structure from ssids
    struct spral_ssids_inform ssids_inform;

    /// \brief
    /// the integer and real output arrays from mc61
    ipc_ mc61_info[10];
    /// see mc61_info
    rpc_ mc61_rinfo[15];

    /// \brief
    /// the output structure from mc64
    struct mc64_info mc64_info;

    /// \brief
    /// the output structure from mc68
    struct mc68_info mc68_info;

    /// \brief
    /// the integer output array from mc77
    ipc_ mc77_info[10];

    /// \brief
    /// the real output status from mc77
    rpc_ mc77_rinfo[10];

    /// \brief
    /// the output scalars and arrays from mumps
    ipc_ mumps_error;
    /// see mumps_error
    ipc_ mumps_info[80];
    /// see mumps_error
    rpc_ mumps_rinfo[40];

    /// \brief
    /// the output scalars and arrays from pardiso
    ipc_ pardiso_error;
    /// see pardiso_error
    ipc_ pardiso_IPARM[64];
    /// see pardiso_error
    rpc_ pardiso_DPARM[64];

    /// \brief
    /// the output scalars and arrays from mkl_pardiso
    ipc_ mkl_pardiso_error;
    /// see mkl_pardiso_error
    ipc_ mkl_pardiso_IPARM[64];

    /// \brief
    /// the output flag from pastix
    ipc_ pastix_info;

    /// \brief
    /// the output scalars and arrays from wsmp
    ipc_ wsmp_error;
    /// see wsmp_error
    ipc_ wsmp_iparm[64];
    /// see wsmp_error
    rpc_ wsmp_dparm[64];

    /// \brief
    /// the output flag from MPI routines
    ipc_ mpi_ierr;

    /// \brief
    /// the output flag from LAPACK routines
    ipc_ lapack_error;

};

// *-*-*-*-*-*-*-*-*-*-    S L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sls_initialize( const char solver[],
                     void **data,
                     struct sls_control_type *control,
                     ipc_ *status );

/*!<
 Select solver, set default control values and initialize private data

 @param[in] solver is a one-dimensional array of type char that specifies
    the \link external solver package \endlink
    that should be used to factorize the matrix \f$A\f$. It should be one of
   'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps',
   'pardiso', 'mkl pardiso', 'pastix', 'wsmp', 'potr', 'sytr' or 'pbtr';
   lower or upper case variants are allowed.

 @param[in,out] data  holds private internal data

 @param[out] control is a struct containing control information
              (see sls_control_type)

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
  \li -26. The requested solver is not available.
*/

// *-*-*-*-*-*-*-*-*-    S L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void sls_read_specfile( struct sls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNSLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/sls.pdf for a list of keywords that may be set.

  @param[in,out] control is a struct containing control information
              (see sls_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-    S L S  _ A N A L Y S E _ M A T R I X   -*-*-*-*-*-*-*-

void sls_analyse_matrix( struct sls_control_type *control,
                         void **data,
                         ipc_ *status,
                         ipc_ n,
                         const char type[],
                         ipc_ ne,
                         const ipc_ row[],
                         const ipc_ col[],
                         const ipc_ ptr[] );

/*!<
 Import structural matrix data into internal storage prior to solution

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sls_control_type)

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The import and analysis were conducted succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 or requirement that the matrix type
       must contain the relevant string 'dense', 'coordinate' or 'sparse_by_rows
       has been violated.
  \li -20. The matrix is not positive definite while the solver used
        expected it to be.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.
  \li -32. More than control.max integer factor size words of internal
       integer storage are required for in-core factorization.
  \li -34. The package PARDISO failed; check the solver-specific
       information components inform.pardiso iparm and inform.pardiso_dparm
       along with PARDISO’s documentation for more details.
  \li -35. The package WSMP failed; check the solver-specific information
       components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s
       documentation for more details.
  \li -36. The scaling package HSL MC64 failed; check the solver-specific
       information component inform.mc64_info along with HSL MC64’s
       documentation for more details.
  \li -37. The scaling package MC77 failed; check the solver-specific
       information components inform.mc77 info and inform.mc77_rinfo along
       with MC77’s documentation for more details.
  \li -43. A direct-access file error occurred. See the value of
       inform.ma77_info.flag for more details.
  \li -50. A solver-specific error occurred; check the solver-specific
       information component of inform along with the solver’s
       documentation for more details.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$A\f$.

 @param[in] type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the matrix \f$A\f$. It should be one of 'coordinate',
   'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$A\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in] row is a one-dimensional array of size ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$A\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in] col is a one-dimensional array of size ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$A\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense storage scheme is used, and in this case
   can be NULL.

 @param[in]  ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$A\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

*/

// *-*-*-*-*-*-*-    S L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void sls_reset_control( struct sls_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
*/

//  *-*-*-*-*-*-   S L S _ f a c t o r i z e _ m a t r i x   -*-*-*-*-*-*-*-

void sls_factorize_matrix( void **data,
                           ipc_ *status,
                           ipc_ ne,
                           const rpc_ val[] );

/*!<
 Form and factorize the symmetric matrix \f$A\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The factors were generated succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 or requirement that the matrix type
       must contain the relevant string 'dense', 'coordinate' or 'sparse_by_rows
       has been violated.
  \li -20. The matrix is not positive definite while the solver used
        expected it to be.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.
  \li -32. More than control.max integer factor size words of internal
       integer storage are required for in-core factorization.
  \li -34. The package PARDISO failed; check the solver-specific
       information components inform.pardiso iparm and inform.pardiso_dparm
       along with PARDISO’s documentation for more details.
  \li -35. The package WSMP failed; check the solver-specific information
       components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s
       documentation for more details.
  \li -36. The scaling package HSL MC64 failed; check the solver-specific
       information component inform.mc64_info along with HSL MC64’s
       documentation for more details.
  \li -37. The scaling package MC77 failed; check the solver-specific
       information components inform.mc77 info and inform.mc77_rinfo along
       with MC77’s documentation for more details.
  \li -43. A direct-access file error occurred. See the value of
       inform.ma77_info.flag for more details.
  \li -50. A solver-specific error occurred; check the solver-specific
       information component of inform along with the solver’s
       documentation for more details.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$A\f$.

 @param[in] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$A\f$ in any of the supported storage schemes.
*/

//  *-*-*-*-*-*-*-*-   S L S _ s o l v e _ s y s t e m   -*-*-*-*-*-*-*-*-*-

void sls_solve_system( void **data,
                       ipc_ *status,
                       ipc_ n,
                       rpc_ sol[] );

/*!<
 Solve the linear system \f$Ax=b\f$.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The required solution was obtained.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -34. The package PARDISO failed; check the solver-specific
       information components inform.pardiso iparm and inform.pardiso_dparm
       along with PARDISO’s documentation for more details.
  \li -35. The package WSMP failed; check the solver-specific information
       components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s
       documentation for more details.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    entries in the vectors \f$b\f$ and \f$x\f$.

 @param[in,out] sol is a one-dimensional array of size n and type double.
    On entry, it must hold the vector \f$b\f$. On a successful exit,
    its contains the solution \f$x\f$.
*/

//  *-*-*-*-*-   S L S _ p a r t i a l _ s o l v e _ s y s t e m   -*-*-*-*-*-

void sls_partial_solve_system( const char part[],
                               void **data,
                               ipc_ *status,
                               ipc_ n,
                               rpc_ sol[] );

/*!<
 Given the factorization \f$A = L D U\f$ with \f$U = L^T\f$,
 solve the linear system \f$Mx=b\f$, where \f$M\f$ is one of
 \f$L\f$, \f$D\f$, \f$U\f$ or \f$S = L \sqrt{D}\f$.


 @param[in] part is a one-dimensional array of type char that specifies the
   component \f$M\f$ of the factorization that is to be used.
   It should be one of "L", "D", "U" or "S", and these correspond to the
   parts  \f$L\f$, \f$D\f$, \f$U\f$ and \f$S\f$; lower or upper case
   variants are allowed.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
  \li  0. The required solution was obtained.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -34. The package PARDISO failed; check the solver-specific
       information components inform.pardiso iparm and inform.pardiso_dparm
       along with PARDISO’s documentation for more details.
  \li -35. The package WSMP failed; check the solver-specific information
       components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s
       documentation for more details.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    entries in the vectors \f$b\f$ and \f$x\f$.

 @param[in,out] sol is a one-dimensional array of size n and type double.
    On entry, it must hold the vector \f$b\f$. On a successful exit,
    its contains the solution \f$x\f$.
*/


// *-*-*-*-*-*-*-*-*-*-    S L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sls_information( void **data,
                      struct sls_inform_type *inform,
                      ipc_ *status );

/*!<
  Provide output information

  @param[in,out] data  holds private internal data

  @param[out] inform is a struct containing output information
              (see sls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void sls_terminate( void **data,
                    struct sls_control_type *control,
                    struct sls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control is a struct containing control information
              (see sls_control_type)

  @param[out] inform  is a struct containing output information
              (see sls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example slst.c
   This is an example of how to use the package in conjunction with the
   sparse linear solver \c sils.
   A variety of supported matrix storage formats are illustrated.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example slstf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
