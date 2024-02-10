//* \file galahad_llst.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LLST C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. May 21st 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package llst

  \section llst_intro Introduction

  \subsection llst_purpose Purpose

  Given a real \f$m\f$ by \f$n\f$ matrix \f$A\f$,
  a real \f$n\f$ by \f$n\f$ symmetric diagonally dominant matrix \f$S\f$
  a real \f$m\f$ vector \f$b\f$ and a scalar \f$\Delta>0\f$, this package
  finds a <b> minimizer of the linear least-squares objective function
  \f$\| A x  - b \|_2\f$, where the vector \f$x\f$ is
  required to satisfy the constraint \f$\|x\|_S \leq  \Delta\f$,</b>
  where the \f$S\f$-norm of \f$x\f$ is
  \f$\|x\|_S = \sqrt{x^T S x}\f$.
  This problem commonly occurs as a trust-region subproblem in nonlinear
  least-squares calculations. The package may also be used to solve the
  related problem in which \f$x\f$ is  instead required to satisfy the
  <b>equality constraint \f$\|x\|_{S} = \Delta\f$</b>.
  The matrix \f$S\f$ need not be provided in the commonly-occurring
  \f$\ell_2\f$-trust-region case for which \f$S = I\f$,
  the \f$n\f$ by \f$n\f$   identity matrix.

  Factorization of matrices of the form
  \f[\mbox{(1)}\;\;\; K(\lambda) = \mat{cc}{ \lambda S & A^T \\ A & - I}\f]
\manonly
\n
    (1)     K(lambda) = ( lambda S   A^T )
                        (      A     - I )
\n
\endmanonly
  of scalars \f$\lambda\f$ will be required, so this package is most suited
  for the case where such a factorization may be found efficiently. If
  this is not the case, the GALAHAD package \c LSTR may be preferred.

  \subsection llst_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection llst_date Originally released

  October 2008, C interface May 2023.

  \subsection llst_terminology Terminology

  The required solution \f$x_*\f$ necessarily satisfies the
  optimality condition \f$A^T A x_* + \lambda_* S x_* = A^T b\f$,
  where \f$\lambda_* \geq 0\f$ is a Lagrange
  multiplier corresponding to the constraint \f$\|x\|_S  \leq  \Delta\f$;
  for the equality-constrained problem \f$\|x\|_S = \Delta\f$ and
  the multiplier is unconstrained.

  \subsection llst_method Method

  The method is iterative, and proceeds in two phases.  Firstly,
  lower and upper bounds, \f$\lambda_L\f$ and
  \f$\lambda_U\f$, on \f$\lambda_*\f$ are computed using
  Gershgorin's theorems and other eigenvalue bounds, including those that
  may involve the Cholesky factorization of \f$S\f$. The first phase of
  the computation proceeds by progressively shrinking the bound interval
  \f$[\lambda_L,\lambda_U]\f$
  until a value \f$\lambda\f$ for which \f$\|x(\lambda)\|_S \geq \Delta\f$
  is found.  Here \f$x(\lambda)\f$ and its companion \f$y(\lambda)\f$ are
  defined to be a solution of
\manonly
  \f[\mbox{(2)}\;\;\;
  (A^T A  + \lambda S)x(\lambda) = A^T b;\f]
\n
   (2)    (A^T A  + \lambda S)x(\lambda) = A^T b;
\n
\endmanonly
  along the way the possibility that \f$\|x(0)\|_S \leq \Delta\f$ is examined,
  and if this transpires the process is terminated with \f$x_* = x(0)\f$.
  Once the terminating \f$\lambda\f$ from the first phase has
  been discovered, the second phase consists of applying Newton or
  higher-order iterations to the nonlinear ``secular'' equation
  \f$\|x(\lambda)\|_S = \Delta\f$ with the knowledge that such
  iterations are both globally and ultimately rapidly convergent.

  The dominant cost is the requirement that we solve a sequence of linear
  systems (2). This may be rewritten as
  \f[\mbox{(3)}\;\;\; \mat{cc}{ \lambda S & A^T \\ A & - I}
  \vect{x(\lambda) \\ y(\lambda)} = \vect{ A^T b \\ 0}\f]
\manonly
\n
    (3)     ( lambda S   A^T ) ( x(lambda) ) = ( A^T b )
            (      A     - I ) ( y(lambda) )   (   0   )
\n
\endmanonly
  for some auxiliary vector \f$y(\lambda)\f$.
  In general a sparse symmetric, indefinite factorization of the
  coefficient matrix \f$K(\lambda)\f$ of (3) is often preferred to a
  Cholesky factorization of that of (2).

  \subsection llst_references Reference

  The method is the obvious adaptation to the linear least-squares
  problem of that described in detail in

  H. S. Dollar, N. I. M. Gould and D. P. Robinson.
  On solving trust-region and other regularised subproblems in optimization.
  Mathematical Programming Computation <b>2(1)</b> (2010) 21--57.

  \subsection llst_call_order Call order

  To solve a given problem, functions from the llst package must be called
  in the following order:

  - \link llst_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link llst_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link llst_import \endlink - set up problem data structures and fixed
      values
  - \link llst_import_scaling \endlink (optional) - set up problem data
      structures for \f$S\f$ if required
  - \link llst_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link llst_solve_problem \endlink - solve the trust-region problem
  - \link llst_information \endlink (optional) - recover information about
    the solution and solution process
  - \link llst_terminate \endlink - deallocate data structures

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

  Likewise, the non-trivial symmetric \f$n\f$ by \f$n\f$ scaling matrix
  \f$S\f$ may be presented
  and stored in a variety of formats. But crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$S\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$S\f$ is symmetric, only the lower triangular part (that is the part
  \f$s_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array S_val
  will hold the value \f$s_{ij}\f$ (and, by symmetry, \f$s_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$S\f$,
  its row index i, column index j
  and value \f$s_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays S_row and
  S_col and real array S_val, respectively, while the number of nonzeros
  is recorded as S_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$S\f$ the i-th component of the
  integer array S_ptr holds the position of the first entry in this row,
  while S_ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$s_{ij}\f$ of the  entries in the i-th row are stored in components
  l = S_ptr(i), \f$\ldots\f$, S_ptr(i+1)-1 of the
  integer array S_col, and real array S_val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsubsection symmetric_matrix_diagonal Diagonal storage format

  If \f$S\f$ is diagonal (i.e., \f$s_{ij} = 0\f$ for all
  \f$0 \leq i \neq j \leq n-1\f$) only the diagonals entries
  \f$s_{ii}\f$, \f$0 \leq i \leq n-1\f$ need
  be stored, and the first n components of the array S_val may be
  used for the purpose.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_LLST_H
#define GALAHAD_LLST_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sbls.h"
#include "galahad_sls.h"
#include "galahad_ir.h"

/**
 * control derived type as a C struct
 */
struct llst_control_type {

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
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// how much of \f$A\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_a;

    /// \brief
    /// how much of \f$S\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_s;

    /// \brief
    /// the maximum number of factorizations (=iterations) allowed.
    /// -ve implies no limit
    ipc_ max_factorizations;

    /// \brief
    /// maximum degree of Taylor approximant allowed (<= 3)
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
    /// stop when \f$| \|x\| -\f$ radius \f$| \leq\f$
    /// max( stop_normal * max( 1, radius )
    rpc_ stop_normal;

    /// \brief
    /// is the solution is <b<required</b> to lie on the boundary
    /// (i.e., is the constraint an equality)?
    bool equality_problem;

    /// \brief
    /// ignore initial_multiplier?
    bool use_initial_multiplier;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

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
    /// control parameters for the symmetric factorization and related linear
    /// solves
    /// (see sbls_c documentation)
    struct sbls_control_type sbls_control;

    /// \brief
    /// control parameters for the factorization of S and related linear solves
    /// (see sls_c documentation)
    struct sls_control_type sls_control;

    /// \brief
    /// control parameters for iterative refinement for definite system solves
    /// (see ir_c documentation)
    struct ir_control_type ir_control;
};

/**
 * time derived type as a C struct
 */
struct llst_time_type {

    /// \brief
    /// total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// CPU time assembling \f$K(\lambda)\f$ in (1)
    rpc_ assemble;

    /// \brief
    /// CPU time spent analysing \f$K(\lambda)\f$
    rpc_ analyse;

    /// \brief
    /// CPU time spent factorizing \f$K(\lambda)\f$
    rpc_ factorize;

    /// \brief
    /// CPU time spent solving linear systems inolving \f$K(\lambda)\f$
    rpc_ solve;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time assembling \f$K(\lambda)\f$
    rpc_ clock_assemble;

    /// \brief
    /// clock time spent analysing \f$K(\lambda)\f$
    rpc_ clock_analyse;

    /// \brief
    /// clock time spent factorizing \f$K(\lambda)\f$
    rpc_ clock_factorize;

    /// \brief
    /// clock time spent solving linear systems inolving \f$K(\lambda)\f$
    rpc_ clock_solve;
};

/**
 * history derived type as a C struct
 */
struct llst_history_type {

    /// \brief
    /// the value of \f$\lambda\f$
    rpc_ lambda;

    /// \brief
    /// the corresponding value of \f$\|x(\lambda)\|_S\f$
    rpc_ x_norm;

    /// \brief
    /// the corresponding value of \f$\|A x(\lambda) - b\|_2\f$
    rpc_ r_norm;
};

/**
 * inform derived type as a C struct
 */
struct llst_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0 the solution has been found
    /// \li -1 an array allocation has failed
    /// \li -2 an array deallocation has failed
    /// \li -3 n and/or Delta is not positive
    /// \li -10 the factorization of \f$K(\lambda)\f$ failed
    /// \li -15 \f$S\f$ does not appear to be strictly diagonally dominant
    /// \li -16 ill-conditioning has prevented furthr progress
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// the number of factorizations performed
    ipc_ factorizations;

    /// \brief
    /// the number of (\f$\|x\|_S\f$,\f$\lambda\f$) pairs in the history
    ipc_ len_history;

    /// \brief
    /// corresponding value of the two-norm of the residual,
    /// \f$\|A x(\lambda) - b\|\f$
    rpc_ r_norm;

    /// \brief
    /// the S-norm of x, \f$\|x\|_S\f$
    rpc_ x_norm;

    /// \brief
    /// the Lagrange multiplier corresponding to the trust-region constraint
    rpc_ multiplier;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// time information
    struct llst_time_type time;

    /// \brief
    /// history information
    struct llst_history_type history[100];

    /// \brief
    /// information from the symmetric factorization and related linear solves
    /// (see sbls_c documentation)
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// information from the factorization of S and related linear solves
    /// (see sls_c documentation)
    struct sls_inform_type sls_inform;

    /// \brief
    /// information from the iterative refinement for definite system solves
    /// (see ir_c documentation)
    struct ir_inform_type ir_inform;
};

// *-*-*-*-*-*-*-*-*-*-    L L S T  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void llst_initialize( void **data,
                     struct llst_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see llst_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L L S T  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void llst_read_specfile( struct llst_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information
              (see llst_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    L L S T  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void llst_import( struct llst_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ m,
                 ipc_ n,
                 const char A_type[],
                 ipc_ A_ne,
                 const ipc_ A_row[],
                 const ipc_ A_col[],
                 const ipc_ A_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see llst_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    residuals, i.e., the number of rows of \f$A\f$. m must be positive.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables, i.e., the number of columns of \f$A\f$. n must be positive.

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

// *-*-*-*-*-*-*-*-    L L S T  _ I M P O R T _ S C A L I N G -*-*-*-*-*-*-*-

void llst_import_scaling( struct llst_control_type *control,
                          void **data,
                          ipc_ *status,
                          ipc_ n,
                          const char S_type[],
                          ipc_ S_ne,
                          const ipc_ S_row[],
                          const ipc_ S_col[],
                          const ipc_ S_ptr[] );

/*!<
 Import the scaling matrix \f$S\f$ into internal storage prior to solution.
 Thus must have been preceeded by a call to llst_import.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see llst_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows' or
       'diagonal' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables, i.e., the number of rows and columns of \f$S\f$.
    n must be positive.

 @param[in]  S_type is a one-dimensional array of type char that specifies the
   \link main_symmetris_matrices symmetric storage scheme \endlink
   used for the matrix \f$S\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense' or 'diagonal'; lower or upper
   case variants are allowed.

 @param[in]  S_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$S\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  S_row is a one-dimensional array of size S_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$S\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  S_col is a one-dimensional array of size S_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$S\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense, diagonal or (scaled) identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  S_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$S\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
*/

// *-*-*-*-*-*-*-    L L S T  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void llst_reset_control( struct llst_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see llst_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  -*-*-*-*-*-*-*-*-*-   L L S T _ S O L V E _ P R O B L E M  -*-*-*-*-*-*-*-*-*-

void llst_solve_problem( void **data,
                         ipc_ *status,
                         ipc_ m,
                         ipc_ n,
                         const rpc_ radius,
                         ipc_ A_ne,
                         const rpc_ A_val[],
                         const rpc_ b[],
                         rpc_ x[],
                         ipc_ S_ne,
                         const rpc_ S_val[] );

/*!<
 Solve the trust-region problem.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
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
  \li -3. The restrictions n > 0 and m > 0 or requirement that A_type or
       A_type contains its relevant string 'dense', 'coordinate',
       'sparse_by_rows' or 'diagonal' has been violated.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -15. The matrix \f$S\f$ does not appear to be strictly diagonally
           dominant.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -17. The step is too small to make further impact.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    residuals

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] radius is a scalar of type rpc_, that
    holds the trust-region radius, \f$\Delta\f$, used. radius must be
    strictly positive

 @param[in] A_ne is a scalar variable of type ipc_, that holds the number of
    entries in the observation matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of size A_ne and type rpc_,
    that holds the values of the entries of the observation matrix
    \f$A\f$ in any of the available storage schemes.

 @param[in] b is a one-dimensional array of size m and type rpc_, that
    holds the values \f$b\f$ of observations. The i-th component
    of b, i = 0, ... , m-1, contains \f$b_i\f$.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in] S_ne is a scalar variable of type ipc_, that holds the number of
    entries in the scaling matrix \f$S\f$ if it not the identity matrix.

 @param[in] S_val is a one-dimensional array of size S_ne and type rpc_,
    that holds the values of the entries of the scaling matrix
    \f$S\f$ in any of the available storage schemes.
    If S_val is NULL, \f$S\f$ will be taken to be the identity matrix.

*/

// *-*-*-*-*-*-*-*-*-*-    L L S T  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void llst_information( void **data,
                       struct llst_inform_type *inform,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see llst_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L L S T  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void llst_terminate( void **data,
                     struct llst_control_type *control,
                     struct llst_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see llst_control_type)

  @param[out] inform   is a struct containing output information
              (see llst_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example llstt.c
   This is an example of how to use the package.\n

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example llsttf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
