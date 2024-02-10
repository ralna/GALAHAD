//* \file galahad_trs.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_TRS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. December 12th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package trs

  \section trs_intro Introduction

  \subsection trs_purpose Purpose

  Given real \f$n\f$ by \f$n\f$ symmetric matrices \f$H\f$ and \f$M\f$
  (with \f$M\f$ diagonally dominant), another real \f$m\f$ by \f$n\f$
  matrix \f$A\f$, a real \f$n\f$ vector \f$c\f$ and scalars
  \f$\Delta>0\f$ and \f$f\f$, this package finds a
  <b>global minimizer of the quadratic objective function
 \f$\frac{1}{2} x^T H  x + c^T x + f\f$,
  where the vector \f$x\f$ is required to satisfy
  the constraint \f$\|x\|_M \leq \Delta\f$ and possibly \f$A x =  0\f$,</b>
  and where the \f$M\f$-norm of \f$x\f$ is \f$\|x\|_M = \sqrt{x^T M x}\f$.
  This problem commonly occurs as a  trust-region subproblem in nonlinear
  optimization calculations.  The package may also be used to solve the
  related problem in which \f$x\f$ is
  instead required to satisfy the <b>equality constraint
  \f$\|x\|_M = \Delta\f$</b>.  The matrix \f$M\f$ need not be provided in
  the commonly-occurring \f$\ell_2\f$-trust-region case for which
  \f$M = I\f$, the \f$n\f$ by \f$n\f$ identity matrix.

  Factorization of matrices of the form \f$H + \lambda M\f$---or
  \f[\mbox{(1)}\;\;\; \mat{cc}{ H + \lambda M & A^T \\ A & 0}\f]
\manonly
\n
    (1)     ( H + lambda M   A^T )
            (      A          0  )
\n
\endmanonly
  in cases where \f$A x = 0\f$ is imposed---for a succession of
  scalars \f$\lambda\f$ will be required, so this package is most suited for
  the case where such a factorization may be found efficiently. If this is
  not the case, the GALAHAD package \c GLTR may be preferred.

  \subsection trs_authors Authors

  N. I. M. Gould and H. S. Thorne, STFC-Rutherford Appleton Laboratory, England,
  and D. P. Robinson, Oxford University, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban,
  Polytechnique Montréal.

  \subsection trs_date Originally released

  October 2008, C interface December 2021.

  \subsection trs_method Method

  The method is iterative, and proceeds in two phases.  Firstly,
  lower and upper bounds, \f$\lambda_L\f$ and
  \f$\lambda_U\f$, on \f$\lambda_*\f$ are computed using
  Gershgorin's theorems and other eigenvalue bounds. The first phase of
  the computation proceeds by progressively shrinking the bound interval
  \f$[\lambda_L,\lambda_U]\f$
  until a value \f$\lambda\f$ for which \f$\|x(\lambda)\|_M \geq \Delta\f$
  is found.  Here \f$x(\lambda)\f$ and its companion \f$y(\lambda)\f$ are
  defined to be a solution of
  \f[\mbox{(2)}\;\;\; (H + \lambda M)x(\lambda)
   + A^T y(\lambda) = - c \;\mbox{and}\; A x(\lambda) = 0;\f]
\manonly
\n
   (2)    (H + lambda M)x(lambda) + A^T y(lambda) = - c and
                      A x(lambda) = 0;
\n
\endmanonly
  along the way the possibility that \f$H\f$ might be positive definite on
  the null-space of \f$A\f$ and \f$\|x(0)\|_M \leq \Delta\f$ is examined, and
  if this transpires the process is terminated with \f$x_* = x(0)\f$.
  Once the terminating \f$\lambda\f$ from the first phase has
  been discovered, the second phase consists of applying Newton or
  higher-order iterations to the nonlinear ``secular'' equation
  \f$\|x(\lambda)\|_M = \Delta\f$ with the knowledge that such
  iterations are both globally and ultimately rapidly convergent. It is
  possible in the ``hard'' case that the interval in the first-phase will
  shrink to the single point \f$\lambda_*\f$, and precautions are taken, using
  inverse iteration with Rayleigh-quotient acceleration to ensure that
  this too happens rapidly.

  The dominant cost is the requirement that we solve a sequence of linear
  systems (2). In the absence of linear constraints, an efficient
  sparse Cholesky factorization with precautions to detect indefinite
  \f$H + \lambda M\f$ is used. If \f$A x = 0\f$ is required, a sparse
  symmetric, indefinite factorization of (1) is used rather than a
  Cholesky factorization.

  \subsection trs_references Reference

  The method is described in detail in

  H. S. Dollar, N. I. M. Gould and D. P. Robinson.
  On solving trust-region and other regularised subproblems in optimization.
  Mathematical Programming Computation <b>2(1)</b> (2010) 21--57.

  \subsection trs_call_order Call order

  To solve a given problem, functions from the trs package must be called
  in the following order:

  - \link trs_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link trs_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link trs_import \endlink - set up problem data structures and fixed
      values
  - \link trs_import_m \endlink - (optional) set up problem data structures
      and fixed values for the scaling matrix \f$M\f$, if any
  - \link trs_import_a \endlink - (optional) set up problem data structures
      and fixed values for the constraint matrix \f$A\f$, if any
  - \link trs_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link trs_solve_problem \endlink - solve the trust-region problem
  - \link trs_information \endlink (optional) - recover information about
    the solution and solution process
  - \link trs_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \subsection main_unsymmetric_matrices Unsymmetric matrix storage formats

  The unsymmetric \f$m\f$ by \f$n\f$ constraint matrix \f$A\f$ may be presented
  and stored in a variety of convenient input formats.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense Dense storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays A_row and
  A_col and real array A_val, respectively, while the number of nonzeros
  is recorded as A_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A\f$ the i-th component of the
  integer array A_ptr holds the position of the first entry in this row,
  while A_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array A_col, and real array A_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  Likewise, the symmetric \f$n\f$ by \f$n\f$ objective Hessian matrix
  \f$H\f$ and scaling matrix \f$M\f$ may be presented
  and stored in a variety of formats. But crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).
  In what follows, we refer to \f$H\f$ but this applies equally to \f$M\f$.

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$H\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$h_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array H_val
  will hold the value \f$h_{ij}\f$ (and, by symmetry, \f$h_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j
  and value \f$h_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays H_row and
  H_col and real array H_val, respectively, while the number of nonzeros
  is recorded as H_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$H\f$ the i-th component of the
  integer array H_ptr holds the position of the first entry in this row,
  while H_ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$h_{ij}\f$ of the  entries in the i-th row are stored in components
  l = H_ptr(i), \f$\ldots\f$, H_ptr(i+1)-1 of the
  integer array H_col, and real array H_val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsubsection symmetric_matrix_diagonal Diagonal storage format

  If \f$H\f$ is diagonal (i.e., \f$H_{ij} = 0\f$ for all
  \f$0 \leq i \neq j \leq n-1\f$) only the diagonals entries
  \f$H_{ii}\f$, \f$0 \leq i \leq n-1\f$ need
  be stored, and the first n components of the array H_val may be
  used for the purpose.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_TRS_H
#define GALAHAD_TRS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "galahad_ir.h"

/**
 * control derived type as a C struct
 */
struct trs_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    ipc_ error;

    /// \brief
    /// unit for monitor output
    ipc_ out;

    /// \brief
    /// unit to write problem data into file problem_file
    ipc_ problem;

    /// \brief
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// should the problem be solved by dense factorization?
    /// Possible values are
    /// \li 0 sparse factorization will be used
    /// \li 1 dense factorization will be used
    /// \li   other the choice is made automatically depending on the dimension
    ///       & sparsity
    ipc_ dense_factorization;

    /// \brief
    /// how much of \f$H\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_h;

    /// \brief
    /// how much of \f$M\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_m;

    /// \brief
    /// how much of \f$A\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_a;

    /// \brief
    /// the maximum number of factorizations (=iterations) allowed.
    /// -ve implies no limit
    ipc_ max_factorizations;

    /// \brief
    /// the number of inverse iterations performed in the "maybe hard" case
    ipc_ inverse_itmax;

    /// \brief
    /// maximum degree of Taylor approximant allowed
    ipc_ taylor_max_degree;

    /// \brief
    /// initial estimate of the Lagrange multipler
    rpc_ initial_multiplier;

    /// \brief
    /// lower and upper bounds on the multiplier, if known
    rpc_ lower;
    /// see lower
    rpc_ upper;

    /// \brief
    /// stop when \f$| ||x|| - radius | \leq\f$
    /// max( stop_normal * radius, stop_absolute_normal )
    rpc_ stop_normal;
    /// see stop_normal
    rpc_ stop_absolute_normal;

    /// \brief
    /// stop when bracket on optimal multiplier \f$\leq\f$ stop_hard *
    ///   max( bracket ends )
    rpc_ stop_hard;

    /// \brief
    /// start inverse iteration when bracket on optimal multiplier \f$\leq\f$
    /// stop_start_invit_tol * max( bracket ends )
    rpc_ start_invit_tol;

    /// \brief
    /// start full inverse iteration when bracket on multiplier \f$\leq\f$
    /// stop_start_invitmax_tol * max( bracket ends)
    rpc_ start_invitmax_tol;

    /// \brief
    /// is the solution is <b<required</b> to lie on the boundary
    /// (i.e., is the constraint an equality)?
    bool equality_problem;

    /// \brief
    /// ignore initial_multiplier?
    bool use_initial_multiplier;

    /// \brief
    /// should a suitable initial eigenvector should be chosen or should a
    /// previous eigenvector may be used?
    bool initialize_approx_eigenvector;

    /// \brief
    /// ignore the trust-region if \f$H\f$ is positive definite
    bool force_Newton;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// name of file into which to write problem data
    char problem_file[31];

    /// \brief
    /// symmetric (indefinite) linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// definite linear equation solver
    char definite_linear_solver[31];

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for the Cholesky factorization and solution
    /// (see sls_c documentation)
    struct sls_control_type sls_control;

    /// \brief
    /// control parameters for iterative refinement (see ir_c documentation)
    struct ir_control_type ir_control;
};

/**
 * time derived type as a C struct
 */
struct trs_time_type {

    /// \brief
    /// total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// CPU time spent building \f$H + \lambda M\f$
    rpc_ assemble;

    /// \brief
    /// CPU time spent reordering \f$H + \lambda M\f$ prior to factorization
    rpc_ analyse;

    /// \brief
    /// CPU time spent factorizing \f$H + \lambda M\f$
    rpc_ factorize;

    /// \brief
    /// CPU time spent solving linear systems inolving \f$H + \lambda M\f$
    rpc_ solve;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time spent building \f$H + \lambda M\f$
    rpc_ clock_assemble;

    /// \brief
    /// clock time spent reordering \f$H + \lambda M\f$ prior to factorization
    rpc_ clock_analyse;

    /// \brief
    /// clock time spent factorizing \f$H + \lambda M\f$
    rpc_ clock_factorize;

    /// \brief
    /// clock time spent solving linear systems inolving \f$H + \lambda M\f$
    rpc_ clock_solve;
};

/**
 * history derived type as a C struct
 */
struct trs_history_type {

    /// \brief
    /// the value of \f$\lambda\f$
    rpc_ lambda;

    /// \brief
    /// the corresponding value of \f$\|x(\lambda)\|_M\f$
    rpc_ x_norm;
};

/**
 * inform derived type as a C struct
 */
struct trs_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0 the solution has been found
    /// \li -1 an array allocation has failed
    /// \li -2 an array deallocation has failed
    /// \li -3 n and/or Delta is not positive
    /// \li -9 the analysis phase of the factorization of
    ///        \f$H + \lambda M\f$ failed
    /// \li -10 the factorization of \f$H + \lambda M\f$ failed
    /// \li -15 \f$M\f$ does not appear to be strictly diagonally dominant
    /// \li -16 ill-conditioning has prevented further progress
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// the number of factorizations performed
    ipc_ factorizations;

    /// \brief
    /// the maximum number of entries in the factors
    int64_t max_entries_factors;

    /// \brief
    /// the number of \f$(||x||_M,\lambda)\f$ pairs in the history
    ipc_ len_history;

    /// \brief
    /// the value of the quadratic function
    rpc_ obj;

    /// \brief
    /// the \f$M\f$-norm of \f$x\f$, \f$||x||_M\f$
    rpc_ x_norm;

    /// \brief
    /// the Lagrange multiplier corresponding to the trust-region constraint
    rpc_ multiplier;

    /// \brief
    /// a lower bound max\f$(0,-\lambda_1)\f$, where \f$\lambda_1\f$
    /// is the left-most eigenvalue of \f$(H,M)\f$
    rpc_ pole;

    /// \brief
    /// was a dense factorization used?
    bool dense_factorization;

    /// \brief
    /// has the hard case occurred?
    bool hard_case;

    /// \brief
    /// name of array that provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// time information
    struct trs_time_type time;
    /// history information
    struct trs_history_type history[100];
    ///  cholesky information (see sls_c documentation)
    struct sls_inform_type sls_inform;
    /// iterative_refinement information (see ir_c documentation)
    struct ir_inform_type ir_inform;
};

// *-*-*-*-*-*-*-*-*-*-    T R S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void trs_initialize( void **data,
                     struct trs_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see trs_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    T R S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void trs_read_specfile( struct trs_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNTRS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/trs.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see trs_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    T R S _ I M P O R T   -*-*-*-*-*-*-*-*-*-*-

void trs_import( struct trs_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 const char H_type[],
                 ipc_ H_ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see trs_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful
  \li -1. An allocation error occurred. A message indicating the
       offending array is written on unit control.error, and the
       returned allocation status and a string containing the name
       of the offending array are held in inform.alloc_status and
       inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the
       offending array is written on unit control.error and the
       returned allocation status and a string containing the
       name of the offending array are held in
       inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type
       contains its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       diagonal' or 'identity' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows (and columns) of H.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian, \f$H\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants
   are allowed.

 @param[in]  H_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$H\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  H_row is a one-dimensional array of size H_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$H\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  H_col is a one-dimensional array of size H_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$H\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense or diagonal storage schemes are used,
   and in this case can be NULL.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    T R S _ I M P O R T _ M  -*-*-*-*-*-*-*-*-*-*-

void trs_import_m( void **data,
                   ipc_ *status,
                   ipc_ n,
                   const char M_type[],
                   ipc_ M_ne,
                   const ipc_ M_row[],
                   const ipc_ M_col[],
                   const ipc_ M_ptr[] );

/*!<
 Import data for the scaling matrix M into internal storage prior to solution.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful
  \li -1. An allocation error occurred. A message indicating the
       offending array is written on unit control.error, and the
       returned allocation status and a string containing the name
       of the offending array are held in inform.alloc_status and
       inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the
       offending array is written on unit control.error and the
       returned allocation status and a string containing the
       name of the offending array are held in
       inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type
       contains its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       diagonal' or 'identity' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows (and columns) of M.

 @param[in]  M_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the scaling matrix, \f$M\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', or 'diagonal'; lower or upper
   case variants are allowed.

 @param[in]  M_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$M\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  M_row is a one-dimensional array of size M_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$M\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  M_col is a one-dimensional array of size M_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$M\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense, diagonal or identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  M_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$M\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    T R S _ I M P O R T _ A  -*-*-*-*-*-*-*-*-*-*-

void trs_import_a( void **data,
                   ipc_ *status,
                   ipc_ m,
                   const char A_type[],
                   ipc_ A_ne,
                   const ipc_ A_row[],
                   const ipc_ A_col[],
                   const ipc_ A_ptr[] );

/*!<
 Import data for the constraint matrix A into internal storage prior
 to solution.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful
  \li -1. An allocation error occurred. A message indicating the
       offending array is written on unit control.error, and the
       returned allocation status and a string containing the name
       of the offending array are held in inform.alloc_status and
       inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the
       offending array is written on unit control.error and the
       returned allocation status and a string containing the
       name of the offending array are held in
       inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type
       contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
       has been violated.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints, i.e., the number of rows of A, if any.
    m must be non-negative.

 @param[in]  A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the constraint Jacobian, \f$A\f$ if any. It should be one of
  'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants
  are allowed.

 @param[in]  A_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$, if used, in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes.

 @param[in]  A_row is a one-dimensional array of size A_ne and type ipc_, that
   holds the row indices of \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes,
   and in this case can be NULL.

 @param[in]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of \f$A\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  A_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each row of \f$A\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.
*/

//  *-*-*-*-*-*-*-*-*-   T R S _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void trs_reset_control( struct trs_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see trs_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-   T R S _ S O L V E _ P R O B L E M   -*-*-*-*-*-*-*-*-*-

void trs_solve_problem( void **data,
                        ipc_ *status,
                        ipc_ n,
                        const rpc_ radius,
                        const rpc_ f,
                        const rpc_ c[],
                        ipc_ H_ne,
                        const rpc_ H_val[],
                        rpc_ x[],
                        ipc_ M_ne,
                        const rpc_ M_val[],
                        ipc_ m,
                        ipc_ A_ne,
                        const rpc_ A_val[],
                        rpc_ y[] );

/*!<
 Solve the trust-region problem.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
  \li  0. The run was succesful.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0, radius > 0 and m > 0 or
       requirement that a type contains its relevant string 'dense',
       'coordinate', 'sparse_by_rows', 'diagonal' or 'identity' has
       been violated.
  \li -9. The analysis phase of the factorization of the matrix (1) failed.
  \li -10. The factorization of the matrix (1) failed.
  \li -15. The matrix M appears not to be diagonally dominant.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -18. Too many factorizations have been required. This may happen if
           control.max factorizations is too small, but may also be
           symptomatic of a badly scaled problem.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] radius is a scalar of type rpc_, that
    holds the trust-region radius, \f$\Delta\f$, used. radius must be
    strictly positive

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ of the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

  @param[in] H_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

  @param[in] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in] M_ne is a scalar variable of type ipc_, that holds the number of
    entries in the scaling matrix \f$M\f$ if it not the identity matrix.

 @param[in] M_val is a one-dimensional array of size M_ne and type rpc_,
    that holds the values of the entries of the scaling matrix
    \f$M\f$, if it is not the identity matrix, in any of the available
     storage schemes. If M_val is NULL, M will be taken to be the identity
     matrix.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints, if any. m must be non-negative.

 @param[in] A_ne is a scalar variable of type ipc_, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$ if used. A_ne must be
    non-negative.

 @param[in] A_val is a one-dimensional array of size A_ne and type rpc_,
    that holds the values of the entries of the constraint Jacobian matrix
    \f$A\f$, if used, in any of the available storage schemes.
    If A_val is NULL, no constraints will be enforced.

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the equality
    constraints \f$A x = 0\f$ if used. The i-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.
*/

// *-*-*-*-*-*-*-*-*-*-    T R S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void trs_information( void **data,
                      struct trs_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see trs_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    T R S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void trs_terminate( void **data,
                    struct trs_control_type *control,
                    struct trs_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see trs_control_type)

  @param[out] inform   is a struct containing output information
              (see trs_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example trst.c
   This is an example of how to use the package to solve a trust-region
   problem. A variety of supported Hessian, scaling and constraint
   matrix storage formats are shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example trstf.c
   This is the same example, but now fortran-style indexing is used.\n

 */


// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
