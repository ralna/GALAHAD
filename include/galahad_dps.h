//* \file galahad_dps.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_DPS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 28th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package dps

  \section dps_intro Introduction

  \subsection dps_purpose Purpose

  Given a real \f$n\f$ by \f$n\f$ symmetric matrix \f$H\f$, this package
  <b>construct a
  symmetric, positive definite matrix \f$M\f$ so that \f$H\f$
  is diagonal in the norm \f$\|v\|_{M} = \sqrt{v^T M v}\f$
  induced by \f$M\f$</b>. Subsequently the package can be use to
  <b>solve the trust-region subproblem</b>
  \f[\mbox{(1)}\;\; \mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + c^T x
    + f \;\; \mbox{subject to}\;\; \|x\||_{M} \leq \Delta\f]
  or the <b>regularized quadratic problem</b>
  \f[\mbox{(2)}\;\;\mbox{minimize}\;\; q(x) + \frac{1}{p} \sigma \|x\||_{M}^p\hspace{50mm} \mbox{$$}\f]
  for a real \f$n\f$ vector \f$c\f$ and scalars \f$f\f$,
  \f$\Delta>0\f$, \f$\sigma>0\f$ and \f$p \geq 2\f$.

  A factorization of the matrix \f$H\f$ will be required, so this package is
  most suited for the case where such a factorization, either dense or sparse,
  may be found efficiently.

  \subsection dps_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection dps_date Originally released

  August 2011, C interface December 2021.

  \subsection dps_terminology Terminology

  \subsection dps_method Method

  The required solution \f$x_*\f$ necessarily satisfies the optimality
  condition \f$H x_* + \lambda_* M x_* + c = 0\f$,
  where \f$\lambda_* \geq 0\f$ is a Lagrange
  multiplier that corresponds to the constraint
  \f$\|x\|_{M}  \leq  \Delta\f$ in the trust-region case (1),
  and is given by \f$\lambda_* = \sigma \|x_*\|^{p-2}\f$
  for the regularization problem (2).
  In addition \f$H + \lambda_* M\f$ will be positive semi-definite; in
  most instances it will actually be positive definite, but in special
  ``hard'' cases singularity is a possibility.

  The matrix \f$H\f$ is decomposed as
  \f[H = P L D L^T P^T\f]
  by calling the GALAHAD package \c SLS.
  Here \f$P\f$ is a permutation matrix,
  \f$L\f$ is unit lower triangular and \f$D\f$ is block diagonal, with
  blocks of dimension at most two. The spectral decomposition of each diagonal
  block of \f$D\f$ is computed, and each eigenvalue \f$\theta\f$ is replaced by
  \f$\max ( | \theta | , \theta_{\min} ) \f$,
  where \f$\theta_{\min}\f$ is a positive user-supplied value. The resulting
  block diagonal matrix is \f$B\f$, from which we define the
  <b>modified-absolute-value</b>
  \f[M = P L B L^T P^T;\f]
  an alternative due to Goldfarb uses instead the simpler
  \f[M = P L L^T P^T.\f]

  Given the factors of \f$H\f$ (and \f$M\f$), the required solution is
  found by making the change of variables \f$y = B^{1/2} L^T P^T x\f$
  (or \f$y = L^T P^T x\f$ in the Goldfarb case)
  which results in ``diagonal'' trust-region and regularization subproblems,
  whose solution may be easily obtained suing a Newton or higher-order iteration
  of a resulting ``secular'' equation.  If subsequent problems, for which
  \f$H\f$ and \f$c\f$ are unchanged, are to be attempted, the existing
  factorization and solution may easily be exploited.

  The dominant cost is that for the factorization of the symmetric, but
  potentially indefinite, matrix \f$H\f$ using the GALAHAD package \c SLS.

  \subsection dps_references Reference

  The method is described in detail for the trust-region case in

  N. I. M. Gould and J. Nocedal (1998).
  The modified absolute-value factorization for trust-region minimization.
  In ``High Performance Algorithms and Software in Nonlinear Optimization''
  (R. De Leone, A. Murli, P. M. Pardalos and G. Toraldo, eds.),
  Kluwer Academic Publishers, pp. 225-241,

  while the adaptation for the regularization case is obvious. The method used
  to solve the diagonal trust-region and regularization subproblems are as
  given by

  H. S. Dollar, N. I. M. Gould and D. P. Robinson (2010).
  On solving trust-region and other regularised subproblems in optimization.
  Mathematical Programming Computation <b>2(1)</b> 21-57

  with simplifications due to the diagonal Hessian.

  \subsection dps_call_order Call order

  To solve a given problem, functions from the dps package must be called
  in the following order:

  - \link dps_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link dps_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link dps_import \endlink - import control and matrix data structures
  - \link dps_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - one of
    - \link dps_solve_tr_problem \endlink - solve the trust-region problem (1)
    - \link dps_solve_rq_problem \endlink - solve the regularized-quadratic
        problem (2)
  - optionally one of
    - \link dps_resolve_tr_problem \endlink - resolve the trust-region problem
        (1) when the non-matrix data has changed
    - \link dps_resolve_rq_problem \endlink - resolve the regularized-quadratic
        problem (2) when the non-matrix data has changed
  - \link dps_information \endlink (optional) - recover information about
    the solution and solution process
  - \link dps_terminate \endlink - deallocate data structures

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

  The symmetric \f$n\f$ by \f$n\f$ coefficient matrix \f$H\f$ may be presented
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

  The matrix \f$H\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$H_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array val
  will hold the value \f$H_{ij}\f$ (and, by symmetry, \f$H_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j
  and value \f$H_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays row and
  col and real array val, respectively, while the number of nonzeros
  is recorded as ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$H\f$ the i-th component of the
  integer array ptr holds the position of the first entry in this row,
  while ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$H_{ij}\f$ of the  entries in the i-th row are stored in components
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
#ifndef GALAHAD_DPS_H
#define GALAHAD_DPS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"

/**
 * control derived type as a C struct
 */
struct dps_control_type {

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
    /// how much of \f$H\f$ has changed since the previous call.
    /// Possible values are
    /// \li 0  unchanged
    /// \li 1  values but not indices have changed
    /// \li 2  values and indices have changed
    ipc_ new_h;

    /// \brief
    /// maximum degree of Taylor approximant allowed
    ipc_ taylor_max_degree;

    /// \brief
    /// smallest allowable value of an eigenvalue of the block diagonal factor
    /// of \f$H\f$
    rpc_ eigen_min;

    /// \brief
    /// lower and upper bounds on the multiplier, if known
    rpc_ lower;
    /// see lower
    rpc_ upper;

    /// \brief
    /// stop trust-region solution when \f$| ||x||_M - \delta | \leq\f$
    /// max( .stop_normal * delta, .stop_absolute_normal )
    rpc_ stop_normal;
    /// see stop_normal
    rpc_ stop_absolute_normal;

    /// \brief
    /// use the Goldfarb variant of the trust-region/regularization norm rather
    /// than the modified absolute-value version
    bool goldfarb;

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
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for the Cholesky factorization and solution
    struct sls_control_type sls_control;
};

/**
 * time derived type as a C struct
 */
struct dps_time_type {

    /// \brief
    /// total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// CPU time spent reordering H prior to factorization
    rpc_ analyse;

    /// \brief
    /// CPU time spent factorizing H
    rpc_ factorize;

    /// \brief
    /// CPU time spent solving the diagonal model system
    rpc_ solve;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time spent reordering H prior to factorization
    rpc_ clock_analyse;

    /// \brief
    /// clock time spent factorizing H
    rpc_ clock_factorize;

    /// \brief
    /// clock time spent solving the diagonal model system
    rpc_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct dps_inform_type {

    /// \brief
    /// return status. See DPS_solve for details
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// the number of 1 by 1 blocks from the factorization of H that were
    /// modified when constructing \f$M\f$
    ipc_ mod_1by1;

    /// \brief
    /// the number of 2 by 2 blocks from the factorization of H that were
    /// modified when constructing \f$M\f$
    ipc_ mod_2by2;

    /// \brief
    /// the value of the quadratic function
    rpc_ obj;

    /// \brief
    /// the value of the regularized quadratic function
    rpc_ obj_regularized;

    /// \brief
    /// the M-norm of the solution
    rpc_ x_norm;

    /// \brief
    /// the Lagrange multiplier associated with the constraint/regularization
    rpc_ multiplier;

    /// \brief
    /// a lower bound max(0,-lambda_1), where lambda_1 is the left-most
    /// eigenvalue of \f$(H,M)\f$
    rpc_ pole;

    /// \brief
    /// has the hard case occurred?
    bool hard_case;

    /// \brief
    /// name of array that provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// time information
    struct dps_time_type time;

    /// \brief
    /// information from SLS
    struct sls_inform_type sls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    D P S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void dps_initialize( void **data,
                     struct dps_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see dps_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    D P S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void dps_read_specfile( struct dps_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNDPS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/dps.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see dps_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    D P S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void dps_import( struct dps_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 const char H_type[],
                 ipc_ ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[] );


/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see dps_control_type)

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
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
       has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows' or
  'dense'; lower or upper case variants are allowed

 @param[in]  ne is a scalar variable of type ipc_, that holds the number of
   entries in the  lower triangular part of H in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  H_row is a one-dimensional array of size ne and type ipc_, that
   holds the row indices of the lower triangular part of H in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL

 @param[in]  H_col is a one-dimensional array of size ne and type ipc_,
   that holds the column indices of the  lower triangular part of H in either
   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
   be set when the dense or diagonal storage schemes are used, and in this
   case can be NULL

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of H, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL
*/

// *-*-*-*-*-*-*-    D P S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void dps_reset_control( struct dps_control_type *control,
                 void **data,
                 ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see dps_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-   D P S _ S O L V E _ T R _ P R O B L E M   -*-*-*-*-*-*-

void dps_solve_tr_problem( void **data,
                           ipc_ *status,
                           ipc_ n,
                           ipc_ ne,
                           rpc_ H_val[],
                           rpc_ c[],
                           rpc_ f,
                           rpc_ radius,
                           rpc_ x[]  );

/*!<
 Find the global minimizer of the trust-region problem (1).

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
       has been violated.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -40. An error has occured when building the preconditioner.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ in the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value of the
    holds the constant term \f$f\f$ in the objective function.

 @param[in]
   radius is a scalar variable pointer of type rpc_, that holds the value
   of the trust-region radius, \f$\Delta > 0\f$.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

*/

//  *-*-*-*-*-*-*-   D P S _ S O L V E _ R Q _ P R O B L E M   -*-*-*-*-*-*-

void dps_solve_rq_problem( void **data,
                           ipc_ *status,
                           ipc_ n,
                           ipc_ ne,
                           rpc_ H_val[],
                           rpc_ c[],
                           rpc_ f,
                           rpc_ power,
                           rpc_ weight,
                           rpc_ x[]  );

/*!<
 Find the global minimizer of the regularized-quadartic problem (2).

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
       has been violated.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -40. An error has occured when building the preconditioner.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ in the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value of the
    holds the constant term \f$f\f$ in the objective function.

 @param[in]
   weight is a scalar variable pointer of type rpc_, that holds the value
   of the regularization weight, \f$\sigma > 0\f$.

 @param[in]
   power is a scalar variable pointer of type rpc_, that holds the value
   of the regularization power, \f$p \geq 2\f$.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

*/

//  *-*-*-*-*-*-*-   D P S _ R E S O L V E _ T R _ P R O B L E M   -*-*-*-*-*-*-

void dps_resolve_tr_problem( void **data,
                             ipc_ *status,
                             ipc_ n,
                             rpc_ c[],
                             rpc_ f,
                             rpc_ radius,
                             rpc_ x[] );

/*!<
 Find the global minimizer of the trust-region problem (1) if some non-matrix
 components have changed since a call to dps_solve_tr_problem.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
       has been violated.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ in the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value
    of the constant term \f$f\f$ in the objective function.

 @param[in]
   radius is a scalar variable pointer of type rpc_, that holds the value
    of the trust-region radius, \f$\Delta > 0\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

*/

//  *-*-*-*-*-*-*-   D P S _ S O L V E _ R Q _ P R O B L E M   -*-*-*-*-*-*-

void dps_resolve_rq_problem( void **data,
                             ipc_ *status,
                             ipc_ n,
                             rpc_ c[],
                             rpc_ f,
                             rpc_ power,
                             rpc_ weight,
                             rpc_ x[] );

/*!<
 Find the global minimizer of the regularized-quadartic problem (2) if some
 non-matrix components have changed since a call to dps_solve_rq_problem.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ in the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value of the
    holds the constant term \f$f\f$ in the objective function.

 @param[in]
   weight is a scalar variable pointer of type rpc_, that holds the value
    of the regularization weight, \f$\sigma > 0\f$.

 @param[in]
   power is a scalar variable pointer of type rpc_, that holds the value
    of the regularization power, \f$p \geq 2\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

*/

// *-*-*-*-*-*-*-*-*-*-    D P S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void dps_information( void **data,
                      struct dps_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see dps_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    D P S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void dps_terminate( void **data,
                    struct dps_control_type *control,
                    struct dps_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see dps_control_type)

  @param[out] inform   is a struct containing output information
              (see dps_inform_type)
 */

/** \example dpst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
