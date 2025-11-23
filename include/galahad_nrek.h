//* \file galahad_nrek.h */

/*
 * THIS VERSION:  GALAHAD 5.4 - 2025-11-21 AT 15:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NREK C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Hussam Al Daas, Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.4. November 14th 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package nrek

  \section nrek_intro Introduction

  \subsection nrek_purpose Purpose

  Given real \f$n\f$ by \f$n\f$ symmetric matrices \f$H\f$ and \f$S\f$
  (with \f$S\f$ diagonally dominant), another real \f$m\f$ by \f$n\f$
  matrix \f$A\f$, a real \f$n\f$ vector \f$c\f$ and a scalar
  \f$\Delta>0\f$, this package finds a
  <b>global minimizer of the regularized quadratic objective function
 \f$\frac{1}{2} x^T H  x + c^T x + f + sigma/p \|x\|^p\f$,
  where the \f$S\f$-norm of \f$x\f$ is \f$\|x\|_S = \sqrt{x^T S x}\f$.
  This problem commonly occurs as a regularization subproblem in nonlinear
  optimization calculations.
  The matrix \f$S\f$ need not be provided in
  the commonly-occurring \f$\ell_2\f$-trust-region case for which
  \f$S = I\f$, the \f$n\f$ by \f$n\f$ identity matrix.

  Factorization of \f$H\f$ and (if present) \f$H\f$ 
  will be required, so this package is most suited for
  the case where such factorizations may be found efficiently. If this is
  not the case, the GALAHAD package \c GLTR may be preferred.

  \subsection nrek_authors Authors

  H. Al Daas and N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban,
  Polytechnique Montréal.

  \subsection nrek_date Originally released

  November 2025.

  \subsection nrek_method Method

  The required solution \f$x_*\f$ necessarily satisfies the optimality 
  condition \f$H x_* + \lambda_* S x_* + c = 0\f$,
  where \f$\lambda_* \geq 0\f$ is a Lagrange multiplier corresponding 
  to the constraint \f$\|x\|_{S}  \leq  \Delta\f$.
  In addition in all cases, the matrix \f$H + \lambda_* S\f$ will 
  be positive semi-definite; in most instances it will actually be positive 
  definite, but in special ``hard'' cases singularity is a possibility.

  The method is iterative, and is based upon building a solution approximation
  from an orthogonal basis of the evolving extended Krylov subspaces
  \f${\cal K}_{2m+1}(H,c) = \mbox{span}\{c,H^{-1}c,H c,
   H^{-2}c, H^2c,\ldots,H^{-m}c,H^{m}c\}\f$
  as \f$m\f$ increases. The key observations are (i) the manifold of
  solutions to the optimality system
  \[ ( H + \lambda I ) x(\lambda) = - c\]
  as a function of \f$\sigma\f$ is of approximately very low rank, (ii)
  the subspace \f${\cal K}_{2m+1}(H,c)\f$ rapidly gives a very good 
  approximation to this manifold, (iii) it is straightforward to
  build an orthogonal basis of \f${\cal K}_{2m+1}(H,c)\f$
  using short-term recurrences and a single factorization of \f$H\f$, and
  (iv) solutions to the trust-region subproblem restricted to elements
  of the orthogonal subspace may be found very efficiently 
  using effective high-order root-finding methods. The fact that the
  second element in the subspace is \f$H^{-1} c\f$ means that it is easy
  to check for the interior-solution possibility \f$x = - H^{-1} c\f$
  that occurs when such a \f$x\f$ satisfies \f$\|x\| \leq \Delta\f$.
  Coping with general scalings \f$S\f$ is a straightforward extension so long
  as factorization of \f$S\f$ is also possible.

  \subsection nrek_references Reference

  The method is described in detail in

  H. Al Daas and N. I. M. Gould.
  Extended-Krylov-subspace methods for trust-region and norm-regularization 
  subproblems.
  Preprint STFC-P-2025-002, Rutherford Appleton Laboratory, Oxfordshire, England

  \subsection nrek_call_order Call order

  To solve a given problem, functions from the nrek package must be called
  in the following order:

  - \link nrek_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link nrek_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link nrek_import \endlink - set up problem data structures and fixed
      values
  - \link nrek_s_import \endlink - (optional) set up problem data structures
      and fixed values for the scaling matrix \f$M\f$, if any
  - \link nrek_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link nrek_solve_problem \endlink - solve the trust-region problem
  - \link nrek_information \endlink (optional) - recover information about
    the solution and solution process
  - \link nrek_terminate \endlink - deallocate data structures

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

  The symmetric \f$n\f$ by \f$n\f$ objective Hessian matrix
  \f$H\f$ and scaling matrix \f$S\f$ may be presented
  and stored in a variety of formats. Crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).
  In what follows, we refer to \f$H\f$ but this applies equally to \f$S\f$.

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

  \subsubsection symmetric_matrix_scaled_identity Multiples of the identity storage format

  If \f$H\f$ is a multiple of the identity matrix, (i.e., \f$H = \alpha I\f$
  where \f$I\f$ is the n by n identity matrix and \f$\alpha\f$ is a scalar),
  it suffices to store \f$\alpha\f$ as the first component of H_val.

  \subsubsection symmetric_matrix_identity The identity matrix format

  If \f$H\f$ is the identity matrix, no values need be stored.

  \subsubsection symmetric_matrix_zero The zero matrix format

  The same is true if \f$H\f$ is the zero matrix.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_NREK_H
#define GALAHAD_NREK_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "galahad_rqs.h"

/**
 * control derived type as a C struct
 */
struct nrek_control_type {

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
    /// maximum dimension of the extended Krylov space
    ipc_ eks_max;

    /// \brief
    /// maximum iteration count
    ipc_ it_max;

    /// \brief
    /// the constant term, f, in the objective function
    rpc_ f;

    /// \brief
    /// increase factor for subsequent trust-region radii
    rpc_ increase;

    /// \brief
    /// stopping tolerance for the cheaply-computed residual
    /// || ( H + lambda S ) x + c || < stop_residua max( 1, ||c|| )
    rpc_ stop_residual;

    /// \brief
    /// should the incoming Lanczos vectors be re-orthogonalised against the
    /// existing ones (this can be expensive)
    bool reorthogonalize;

    /// \brief
    /// choose between two versions, either that given as Algorithm 5.2  or 
    /// B.3 in the paper, for recurrences when a non-unit S is given
    bool s_version_52;

    /// \brief
    /// make a tiny perturbation to the term c to try to protect from the 
    /// hard case
    bool perturb_c;

    /// \brief
    /// check for convergence for all system orders, not just even ones
    bool stop_check_all_orders;

    /// \brief
    /// resolve a previously solved problem with a larger weight
    bool new_weight;

    /// \brief
    /// solve a problem with the same structure as the previous one but with 
    /// different values of H, c and/or S
    bool new_values;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// linear equation solver for systems with matrix H
    char linear_solver[31];

    /// \brief
    /// linear equation solver for systems with matrix S
    char linear_solver_for_s[31];

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
    /// control parameters for the Cholesky factorization and solution
    /// (see sls_c documentation)
    struct sls_control_type sls_s_control;

    /// \brief
    /// control parameters for diagonal trust-region solves 
    /// (see rqs_c documentation)
    struct rqs_control_type rqs_control;
};

/**
 * time derived type as a C struct
 */
struct nrek_time_type {

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
 * inform derived type as a C struct
 */
struct nrek_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0 the solution has been found
    /// \li -1 an array allocation has failed
    /// \li -2 an array deallocation has failed
    /// \li -3 n and/or Delta is not positive
    /// \li -9 the analysis phase of the factorization of \f$H\f$ or \f$S\f$ 
    ///    failed
    /// \li -10 the factorization of \f$H\f$ or \f$S\f$ failed
    /// \li -11 the solve phase involving \f$H\f$ or \f$S\f$ failed
    /// \li -15 \f$S\f$ does not appear to be strictly diagonally dominant
    /// \li -16 ill-conditioning has prevented further progress
    /// \li -18 too many iterations have been required. This may happen if 
    ///     control.eks max is too small, but may also be symptomatic of a
    ///     badly scaled problem.
    /// \li -31 a resolve call has been made before an initial call (see 
    ///     the argument control.new_weight.
    /// \li -38 an error occurred in a call to an LAPACK subroutine

    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// the number of iterations performed
    ipc_ iter;

    /// \brief
    /// the number of orthogonal vectors required
    ipc_ n_vec;

    /// \brief
    /// the value of the quadratic function
    rpc_ obj;

    /// \brief
    /// the value of the regularized quadratic function
    rpc_ obj_regularized;

    /// \brief
    /// the \f$S\f$-norm of \f$x\f$, \f$||x||_S\f$
    rpc_ x_norm;

    /// \brief
    /// the Lagrange multiplier corresponding to the trust-region constraint
    rpc_ multiplier;

    /// \brief
    /// the current weight
    rpc_ weight;

    /// \brief
    /// the proposed next weight to be used
    rpc_ next_weight;

    /// \brief
    /// the maximum relative residual error
    rpc_  error;

    /// \brief
    /// name of array that provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// time information
    struct nrek_time_type time;
    ///  cholesky information for \f$H\f$ (see sls_c documentation)
    struct sls_inform_type sls_inform;
    ///  cholesky information for \f$S\f$ (see sls_c documentation)
    struct sls_inform_type sls_s_inform;
    /// diagonal trust-region solve information (see rqs_c documentation)
    struct rqs_inform_type rqs_inform;
};

// *-*-*-*-*-*-*-*-*-*-    N R E K  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void nrek_initialize( void **data,
                      struct nrek_control_type *control,
                      ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see nrek_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    N R E K  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void nrek_read_specfile( struct nrek_control_type *control,
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNNREK.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/nrek.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see nrek_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    N R E K _ I M P O R T   -*-*-*-*-*-*-*-*-*-*-

void nrek_import( struct nrek_control_type *control,
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
  paramters for the remaining prcedures (see nrek_control_type)

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

// *-*-*-*-*-*-*-*-*-*-*-*-    N R E K _ S _ I M P O R T  -*-*-*-*-*-*-*-*-*-*-

void nrek_s_import( void **data,
                    ipc_ *status,
                    ipc_ n,
                    const char S_type[],
                    ipc_ S_ne,
                    const ipc_ S_row[],
                    const ipc_ S_col[],
                    const ipc_ S_ptr[] );

/*!<
 Import data for the scaling matrix \f$S\f$ into internal storage prior 
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
  \li -3. The restriction n > 0 or requirement that a type
       contains its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       diagonal' or 'identity' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows (and columns) of \f$S\f$.

 @param[in]  S_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the scaling matrix, \f$S\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', or 'diagonal'; lower or upper
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
   need not be set when the dense, diagonal or identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  S_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$S\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
*/

//  *-*-*-*-*-*-*-*-*-   N R E K _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void nrek_reset_control( struct nrek_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see nrek_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-   N R E K _ S O L V E _ P R O B L E M   -*-*-*-*-*-*-*-*-

void nrek_solve_problem( void **data,
                         ipc_ *status,
                         ipc_ n,
                         ipc_ H_ne,
                         const rpc_ H_val[],
                         const rpc_ c[],
                         const rpc_ power,
                         const rpc_ weight,
                         rpc_ x[],
                         ipc_ S_ne,
                         const rpc_ S_val[] );

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
  \li -3. The restrictions n > 0 or weight > 0 or
       requirement that a type contains its relevant string 'dense',
       'coordinate', 'sparse_by_rows', 'diagonal' or 'identity' has
       been violated.
   \li -9 the analysis phase of the factorization of \f$H\f$ or \f$S\f$ failed
   \li -10 the factorization of \f$H\f$ or \f$S\f$ failed
   \li -11 the solve phase involving \f$H\f$ or \f$S\f$ failed
   \li -15 \f$S\f$ does not appear to be strictly diagonally dominant
   \li -16 ill-conditioning has prevented further progress
   \li -18 too many iterations have been required. This may happen if 
       control.eks max is too small, but may also be symptomatic of a
       badly scaled problem.
   \li -31 a resolve call has been made before an initial call (see 
       the argument control.new_weight.
   \li -38 an error occurred in a call to an LAPACK subroutine

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

  @param[in] H_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

  @param[in] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] c is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$c\f$ of the objective function.
    The j-th component of c, j = 0, ... ,  n-1, contains  \f$c_j \f$.

 @param[in] power is a scalar of type rpc_, that
    holds the regularization power, \f$p\f$, used. weight must be
    strictly larger than two.

 @param[in] weight is a scalar of type rpc_, that
    holds the regularization weight, \f$\sigma\f$, used. weight must be
    strictly positive.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in] S_ne is a scalar variable of type ipc_, that holds the number of
    entries in the scaling matrix \f$S\f$ if it not the identity matrix.

 @param[in] S_val is a one-dimensional array of size S_ne and type rpc_,
    that holds the values of the entries of the scaling matrix
    \f$S\f$, if it is not the identity matrix, in any of the available
    storage schemes. If S_val is NULL, \f$S\f$ will be taken to be the 
    identity matrix.
*/

// *-*-*-*-*-*-*-*-*-*-    N R E K  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void nrek_information( void **data,
                       struct nrek_inform_type *inform,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see nrek_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    N R E K  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void nrek_terminate( void **data,
                     struct nrek_control_type *control,
                     struct nrek_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see nrek_control_type)

  @param[out] inform   is a struct containing output information
              (see nrek_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example nrekt.c
   This is an example of how to use the package to solve a trust-region
   subproblem. A variety of supported Hessian and scaling matrix storage
   formats are shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example nrektf.c
   This is the same example, but now fortran-style indexing is used.\n

 */


// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
