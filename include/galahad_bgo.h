/** \file galahad_bgo.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 3rd 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package bgo

  \section bgo_intro Introduction

  \subsection bgo_purpose Purpose

  The bgo package uses a <b>multi-start trust-region method to find an
  approximation to the global minimizer of a differentiable objective
  function \f$f(x)\f$ of \f$n\f$ variables \f$x\f$, subject to simple bounds
  \f$x^l \leq x \leq x^u\f$ on the variables.</b>
  Here, any of the components of the vectors of bounds \f$x^l\f$ and \f$x^u\f$
  may be infinite. The method offers the choice of direct
  and iterative solution of the key trust-region subproblems, and
  is suitable for large problems. First derivatives are required,
  and if second derivatives can be calculated, they will be exploited---if
  the product of second derivatives with a vector may be found but
  not the derivatives themselves, that may also be exploited.

  The package offers both random multi-start and local-minimize-and probe
  methods to try to locate the global minimizer. There are no theoretical
  guarantees unless the sampling is huge, and realistically the success of
  the methods decreases as the dimension and nonconvexity increase.

  \subsection bgo_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection bgo_date Originally released

  July 2016, C interface August 2021.

  \subsection bgo_terminology Terminology

  The \e gradient \f$\nabla_x f(x)\f$ of \f$f(x)\f$ is the vector whose
  \f$i\f$-th component is \f$\partial f(x)/\partial x_i\f$.
  The \e Hessian \f$\nabla_{xx} f(x)\f$ of \f$f(x)\f$ is the symmetric matrix
  whose \f$i,j\f$-th entry is \f$\partial^2 f(x)/\partial x_i \partial x_j\f$.
  The Hessian is \e sparse if a significant and useful proportion of the
  entries are universally zero.

  \subsection bgo_method Method

  A choice of two methods is available.
  In the first, local-minimization-and-probe, approach, local minimization
  and univariate global minimization are intermixed. Given a current
  champion \f$x^S_k\f$, a local minimizer \f$x_k\f$ of \f$f(x)\f$ within the
  feasible box \f$x^l \leq x \leq x^u\f$ is found using the GALAHAD package trb.
  Thereafter \f$m\f$ random directions \f$p\f$ are generated, and univariate
  local minimizer of \f$f(x_k + \alpha p)\f$ as a function of the scalar
  \f$\alpha\f$ along each \f$p\f$ within the interval \f$[\alpha^L,\alpha^u]\f$,
  where \f$\alpha^L\f$ and \f$\alpha^u\f$ are the smallest and largest
  \f$\alpha\f$ for which \f$x^l \leq x_k + \alpha p \leq x^u\f$,
  is performed using the GALAHAD package ugo. The point \f$x_k + \alpha p\f$
  that gives the smallest value of \f$f\f$ is then selected as the new champion
  \f$x^S_{k+1}\f$.

  The random directions \f$p\f$ are chosen in one of three ways. The simplest is
  to select the components as
\manonly
(ignore next phrase - doxygen bug!)
\endmanonly
\f[p_i = \mbox{pseudo random} \in
\left\{
\begin{array}{rl}
\mbox{[-1,1]} & \mbox{if} \;\; x^l_i < x_{k,i} < x^u_i \\
\mbox{[0,1]} & \mbox{if} \;\; x_{k,i} = x^l_i \\
\mbox{[-1,0]} & \mbox{if} \;\; x_{k,i} = x^u_i
\end{array}
\right.
\f]
\manonly
  \n
                         ( [-1,1] if x^l_i < x_{k,i} < x^u_i
  p_i = pseudo random in ( [0,1] if x_{k,i} = x^l_i
                         ( [-1,0] if x_{k,i} = x^u_i
  \n
\endmanonly
  for each \f$1 \leq i \leq n\f$. An alternative is to
  pick \f$p\f$ by partitioning each dimension of the feasible ``hypercube'' box
  into \f$m\f$ equal segments, and then selecting sub-boxes
  randomly within this hypercube using GALAHAD's Latin hypercube sampling
  package, lhs.
  Each components of \f$p\f$ is then selected in its sub-box, either uniformly
  or pseudo randomly.

  The other, random-multi-start, method provided selects
  \f$m\f$ starting points
  at random, either componentwise pseudo randomly in the feasible box, or by
   partitioning each component into \f$m\f$ equal segments, assigning each to
  a sub-box using Latin hypercube sampling, and finally choosing the
  values either uniformly or pseudo randomly. Local minimizers within the
  feasible box are then computed by the GALAHAD package trb, and
  the best is assigned as the current champion. This process is then
  repeated until evaluation limits are achieved.

  If \f$n=1\f$, the GALAHAD package UGO is called directly.

  We reiterate that there are no theoretical guarantees unless the sampling
  is huge, and realistically the success of the methods decreases as the
  dimension and nonconvexity increase. Thus the methods used should best
  be viewed as heuristics.

  \subsection bgo_references References

  The generic bound-constrained trust-region method is described in detail in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint (2000),
  Trust-region methods.
  SIAM/MPS Series on Optimization,

  the univariate global minimization method employed is an extension of that
  due to

  D. Lera and Ya. D. Sergeyev (2013),
  ``Acceleration of univariate global optimization algorithms working with
  Lipschitz functions and Lipschitz first derivatives''
  SIAM J. Optimization Vol. 23, No. 1, pp. 508–529,

  while the Latin-hypercube sampling method employed is that of

  B. Beachkofski and R. Grandhi (2002).
  ``Improved Distributed Hypercube Sampling'',
  43rd AIAA structures, structural dynamics, and materials conference,
  pp. 2002-1274.

  \section bgo_call_order Call order

  To solve a given problem, functions from the bgo package must be called
  in the following order:

  - \link bgo_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link bgo_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link bgo_import \endlink - set up problem data structures and fixed
      values
  - \link bgo_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link bgo_solve_with_mat \endlink - solve using function calls to
       evaluate function, gradient and Hessian values
     - \link bgo_solve_without_mat \endlink - solve using function calls to
       evaluate function and gradient values and Hessian-vector products
     - \link bgo_solve_reverse_with_mat \endlink - solve returning to the
       calling program to obtain function, gradient and Hessian values, or
     - \link bgo_solve_reverse_without_mat \endlink - solve returning to the
       calling prorgram to obtain function and gradient values and
       Hessian-vector products
  - \link bgo_information \endlink (optional) - recover information about
    the solution and solution process
  - \link bgo_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \section main_symmetric_matrices Symmetric matrix storage formats

  The symmetric \f$n\f$ by \f$n\f$ matrix \f$H = \nabla_{xx}f\f$ may be
  presented and stored in a variety of formats. But crucially symmetry
  is exploited by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  Both C-style (0 based) and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsection symmetric_matrix_dense Dense storage format

  The matrix \f$H\f$ is stored as a compact dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$H_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$ of the storage array H_val
  will hold the value \f$H_{ij}\f$ (and, by symmetry, \f$H_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j
  and value \f$H_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$, are stored as
  the \f$l\f$-th components of the integer arrays H_row and
  H_col and real array H_val, respectively, while the number of nonzeros
  is recorded as H_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$H\f$ the i-th component of the
  integer array H_ptr holds the position of the first entry in this row,
  while H_ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$H_{ij}\f$ of the entries in the i-th row are stored in components
  l = H_ptr(i), \f$\ldots\f$, H_ptr(i+1)-1 of the
  integer array H_col, and real array H_val, respectively.
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
#ifndef GALAHAD_BGO_H
#define GALAHAD_BGO_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_trb.h"
#include "galahad_ugo.h"
#include "galahad_lhs.h"

/*
 * control derived type as a C struct
 */
struct bgo_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required. Possible values are:
    /// \li \f$\leq\f$ 0 no output,
    /// \li 1 a one-line summary for every improvement
    /// \li 2 a summary of each iteration
    /// \li \f$\geq\f$ 3 increasingly verbose (debugging) output
    ipc_ print_level;

    /// \brief
    /// the maximum number of random searches from the best point found so far
    ipc_ attempts_max;

    /// \brief
    /// the maximum number of function evaluations made
    ipc_ max_evals;

    /// \brief
    /// sampling strategy used. Possible values are
    /// \li 1 uniformly spread
    /// \li 2 Latin hypercube sampling
    /// \li 3 niformly spread within a Latin hypercube
    ipc_ sampling_strategy;

    /// \brief
    /// hyper-cube discretization (for sampling stategies 2 and 3)
    ipc_ hypercube_discretization;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    ipc_ alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// the smallest value the objective function may take before the problem
    /// is marked as unbounded
    rpc_ obj_unbounded;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// perform random-multistart as opposed to local minimize and probe
    bool random_multistart;

    /// \brief
    /// is the Hessian matrix of second derivatives available or is access only
    /// via matrix-vector products?
    bool hessian_available;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for UGO
    struct ugo_control_type ugo_control;

    /// \brief
    /// control parameters for LHS
    struct lhs_control_type lhs_control;

    /// \brief
    /// control parameters for TRB
    struct trb_control_type trb_control;
};

/*
 * time derived type as a C struct
 */
struct bgo_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the CPU time spent performing univariate global optimization
    real_sp_ univariate_global;

    /// \brief
    /// the CPU time spent performing multivariate local optimization
    real_sp_ multivariate_local;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent performing univariate global optimization
    rpc_ clock_univariate_global;

    /// \brief
    /// the clock time spent performing multivariate local optimization
    rpc_ clock_multivariate_local;
};

/*
 * inform derived type as a C struct
 */
struct bgo_inform_type {

    /// \brief
    /// return status. See BGO_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of evaluations of the objective function
    ipc_ f_eval;

    /// \brief
    /// the total number of evaluations of the gradient of the objective
    /// function
    ipc_ g_eval;

    /// \brief
    /// the total number of evaluations of the Hessian of the objective function
    ipc_ h_eval;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by BGO_solve
    rpc_ obj;

    /// \brief
    /// the norm of the projected gradient of the objective function at the best
    /// estimate of the solution determined by BGO_solve
    rpc_ norm_pg;

    /// \brief
    /// timings (see above)
    struct bgo_time_type time;

    /// \brief
    /// inform parameters for UGO
    struct ugo_inform_type ugo_inform;

    /// \brief
    /// inform parameters for LHS
    struct lhs_inform_type lhs_inform;

    /// \brief
    /// inform parameters for TRB
    struct trb_inform_type trb_inform;
};

/*  *-*-*-*-*-*-*-*-*-*-   B G O _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*
 *
 * Provide default values for BGO controls
 */

void bgo_initialize( void **data,
                     struct bgo_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

 @param[in,out] data  holds private internal data

 @param[out] control is a struct containing control information
              (see bgo_control_type)

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    B G O  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void bgo_read_specfile( struct bgo_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNBGO.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/bgo.pdf for a list of keywords that may be set.

  @param[in,out] control is a struct containing control information
              (see bgo_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

//  *-*-*-*-*-*-*-*-*-*-*-*-   B G O _ I M P O R T    -*-*-*-*-*-*-*-*-*-*-*-*

void bgo_import( struct bgo_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 const rpc_ x_l[],
                 const rpc_ x_u[],
                 const char H_type[],
                 ipc_ ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bgo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] x_l is a one-dimensional array of size n and type rpc_,
    that holds the values \f$x^l\f$ of the lower bounds on the optimization
    variables \f$x\f$. The j-th component of x_l, \f$j = 0, \ldots, n-1\f$,
    contains \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_,
    that holds the values \f$x^u\f$ of the upper bounds on the optimization
    variables \f$x\f$. The j-th component of x_u, \f$j = 0, \ldots, n-1\f$,
    contains \f$x^u_j\f$.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows',
  'dense', 'diagonal' or 'absent', the latter if access to the Hessian is
  via matrix-vector products; lower or upper case variants are allowed.

 @param[in]  ne is a scalar variable of type ipc_, that holds the number of
   entries in the  lower triangular part of H in the sparse co-ordinate
   storage scheme. It need not be set for any of the other three schemes.

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


//  *-*-*-*-*-*-*-*-*-   B G O _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void bgo_reset_control( struct bgo_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bgo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
 */

//  *-*-*-*-*-*-*-*-*-   B G O _ S O L V E _ W I T H _ M A T   -*-*-*-*-*-*-*-*

void bgo_solve_with_mat( void **data,
                         void *userdata,
                         ipc_ *status,
                         ipc_ n,
                         rpc_ x[],
                         rpc_ g[],
                         ipc_ ne,
                         ipc_ (*eval_f)(
                           ipc_, const rpc_[], rpc_*, const void * ),
                         ipc_ (*eval_g)(
                           ipc_, const rpc_[], rpc_[], const void * ),
                         ipc_ (*eval_h)(
                           ipc_, ipc_, const rpc_[], rpc_[],
                           const void * ),
                         ipc_ (*eval_hprod)(
                           ipc_, const rpc_[], rpc_[], const rpc_[],
                           bool, const void * ),
                         ipc_ (*eval_prec)(
                           ipc_, const rpc_[], rpc_[], const rpc_[],
                           const void * ) );

/*!<
 Find an approximation to the global minimizer of a given function subject to
 simple bounds on the variables using a multistart trust-region method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is
 provided specifically, and all function/derivative information is
 available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] g is a one-dimensional array of size n and type rpc_, that
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param eval_f is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_f( ipc_ n, const rpc_ x[], rpc_ *f, const void *userdata )
   \endcode
   The value of the objective function \f$f(x)\f$ evaluated at x=\f$x\f$ must
   be assigned to f, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_f via the structure \c userdata.

 @param eval_g is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_g( ipc_ n, const rpc_ x[], rpc_ g[], const void *userdata )
   \endcode
   The components of the gradient \f$g = \nabla_x f(x\f$) of the objective
   function evaluated at x=\f$x\f$ must be assigned to g, and the function
   return value set to 0. If the evaluation is impossible at x, return
   should be set to a nonzero value.
   Data may be passed into \c eval_g via the structure \c userdata.

 @param eval_h is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_h( ipc_ n, ipc_ ne, const rpc_ x[], rpc_ h[],
                    const void *userdata )
   \endcode
   The nonzeros of the Hessian \f$H = \nabla_{xx}f(x)\f$ of the objective
   function evaluated at x=\f$x\f$ must be assigned to h in the same order
   as presented to bgo_import, and the function return value set to 0.
   If the evaluation is impossible at x, return should be set to a
   nonzero value.
   Data may be passed into \c eval_h via the structure \c userdata.

 @param  eval_prec is an optional user-supplied function that may be NULL.
   If non-NULL, it must have the following signature:
   \code
       ipc_ eval_prec( ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[],
                      const void *userdata )
   \endcode
   The product \f$u = P(x) v\f$ of the user's preconditioner \f$P(x)\f$
   evaluated at \f$x\f$ with the vector v = \f$v\f$, the result \f$u\f$
   must be retured in u, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata.
 */

//  *-*-*-*-*-*-*-*-   B G O _ S O L V E _ W I T H O U T _ M A T   -*-*-*-*-*-*

void bgo_solve_without_mat( void **data,
                            void *userdata,
                            ipc_ *status,
                            ipc_ n,
                            rpc_ x[],
                            rpc_ g[],
                            ipc_ (*eval_f)(
                              ipc_, const rpc_[], rpc_*, const void * ),
                            ipc_ (*eval_g)(
                              ipc_, const rpc_[], rpc_[], const void * ),
                            ipc_ (*eval_hprod)(
                              ipc_, const rpc_[], rpc_[],
                              const rpc_[], bool, const void * ),
                            ipc_ (*eval_shprod)(ipc_, const rpc_[], ipc_,
                              const ipc_[], const rpc_[], ipc_*, ipc_[],
                              rpc_[], bool, const void * ),
                            ipc_ (*eval_prec)(
                              ipc_, const rpc_[], rpc_[],
                              const rpc_[], const void * ) );

/*!<
 Find an approximation to the global minimizer of a given function subject to
 simple bounds on the variables using a multistart trust-region method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is
 provided by Hessian-vector products, and all function/derivative
 information is available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] g is a one-dimensional array of size n and type rpc_, that
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param eval_f is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_f( ipc_ n, const rpc_ x[], rpc_ *f, const void *userdata )
   \endcode
   The value of the objective function \f$f(x)\f$ evaluated at x=\f$x\f$ must
   be assigned to f, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_f via the structure \c userdata.

 @param eval_g is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_g( ipc_ n, const rpc_ x[], rpc_ g[], const void *userdata )
   \endcode
   The components of the gradient \f$g = \nabla_x f(x\f$) of the objective
   function evaluated at x=\f$x\f$ must be assigned to g, and the function
   return value set to 0. If the evaluation is impossible at x, return
   should be set to a nonzero value.
   Data may be passed into \c eval_g via the structure \c userdata.

 @param eval_hprod is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_hprod( ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[],
                        bool got_h, const void *userdata )
   \endcode
   The sum \f$u + \nabla_{xx}f(x) v\f$ of the product of the Hessian
   \f$\nabla_{xx}f(x)\f$ of the objective function evaluated at x=\f$x\f$
   with the vector v=\f$v\f$ and the vector $\f$u\f$ must be returned in u,
   and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   The Hessian has already been evaluated or used at x if got_h is true.
   Data may be passed into \c eval_hprod via the structure \c userdata.

 @param eval_shprod is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_shprod( ipc_ n, const rpc_ x[], ipc_ nnz_v,
                         const ipc_ index_nz_v[], const rpc_ v[],
                         ipc_ *nnz_u, ipc_ index_nz_u[], rpc_ u[],
                         bool got_h, const void *userdata )
   \endcode
   The product \f$u = \nabla_{xx}f(x) v\f$ of the Hessian
   \f$\nabla_{xx}f(x)\f$ of the objective function evaluated at \f$x\f$
   with the sparse vector v=\f$v\f$ must be returned in u, and the function
   return value set to 0. Only the components index_nz_v[0:nnz_v-1] of v
   are nonzero, and the remaining components may not have been be set.
   On exit, the user must indicate the nnz_u indices of u that are nonzero
   in index_nz_u[0:nnz_u-1], and only these components of u need be set.
   If the evaluation is impossible at x, return should be set to a nonzero
   value. The Hessian has already been evaluated or used at x if got_h is true.
   Data may be passed into \c eval_prec via the structure \c userdata.

 @param  eval_prec is an optional user-supplied function that may be NULL.
   If non-NULL, it must have the following signature:
   \code
       ipc_ eval_prec( ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[],
                      const void *userdata )
   \endcode
   The product \f$u = P(x) v\f$ of the user's preconditioner \f$P(x)\f$
   evaluated at \f$x\f$ with the vector v = \f$v\f$, the result \f$u\f$
   must be retured in u, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata.
 */

//  *-*-*-*-*-   B G O _ S O L V E _ R E V E R S E _ W I T H _ M A T   -*-*-*-*

void bgo_solve_reverse_with_mat( void **data,
                                 ipc_ *status,
                                 ipc_ *eval_status,
                                 ipc_ n,
                                 rpc_ x[],
                                 rpc_ f,
                                 rpc_ g[],
                                 ipc_ ne,
                                 rpc_ H_val[],
                                 const rpc_ u[],
                                 rpc_ v[] );

/*!<
 Find an approximation to the global minimizer of a given function subject to
 simple bounds on the variables using a multistart trust-region method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is
 provided specifically, but function/derivative information is only
 available by returning to the calling procedure

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param status (continued)
  \li  2. The user should compute the objective function value \f$f(x)\f$ at
         the point \f$x\f$ indicated in x and then re-enter the function.
         The required value should be set in f, and eval_status should be
         set to 0. If the user is unable to evaluate \f$f(x)\f$--- for
         instance, if the function is undefined at \f$x\f$--- the user need
         not set f, but should then set eval_status to a non-zero value.
  \li   3. The user should compute the gradient of the objective function
         \f$\nabla_x f(x)\f$ at the point \f$x\f$ indicated in x and then
         re-enter the function. The value of the i-th component of the g
         radient should be set in g[i], for i = 0, ..., n-1 and eval_status
         should be set to 0. If the user is unable to evaluate a component
         of \f$\nabla_x f(x)\f$ --- for instance if a component of the gradient
         is undefined at \f$x\f$ -the user need not set g, but should then set
         eval_status to a non-zero value.
  \li   4. The user should compute the Hessian of the objective function
         \f$\nabla_{xx}f(x)\f$ at the point x indicated in \f$x\f$ and then
         re-enter the function. The value l-th component of the Hessian stored
         according to the scheme input in the remainder of \f$H\f$ should be
         set in H_val[l], for l = 0, ..., ne-1 and eval_status should be set
         to 0. If the user is unable to evaluate a component of
         \f$\nabla_{xx}f(x)\f$ --- for instance, if a component of the Hessian
         is undefined at \f$x\f$ --- the user need not set H_val, but should
         then set eval_status to a non-zero value.
  \li  5. The user should compute the product \f$\nabla_{xx}f(x)v\f$ of the
        Hessian of the objective function \f$\nabla_{xx}f(x)\f$ at the point
        \f$x\f$ indicated in x with the vector \f$v\f$, add the result to
        the vector \f$u\f$ and then re-enter the function. The vectors
        \f$u\f$ and \f$v\f$ are given in u and v respectively, the resulting
        vector \f$u + \nabla_{xx}f(x)v\f$ should be set
        in u and eval_status should be set to 0. If the user is unable to
        evaluate the product--- for instance, if a component of the Hessian is
        undefined at \f$x\f$ --- the user need not alter u, but should then set
        eval_status to a non-zero value.
  \li   6. The user should compute the product \f$u = P(x)v\f$ of their
         preconditioner \f$P(x)\f$ at the point x indicated in \f$x\f$ with
         the vector \f$v\f$ and then re-enter the function. The vector \f$v\f$
         is given in v, the resulting vector \f$u = P(x)v\f$ should be set in
         u and eval_status should be set to 0. If the user is unable to
         evaluate the product--- for instance, if a component of the
         preconditioner is undefined at \f$x\f$ --- the user need not set u,
         but should then set eval_status to a non-zero value.
  \li 23. The user should follow the instructions for 2 <b>and</b> 3
        above before returning.
  \li 25. The user should follow the instructions for 2 <b>and</b> 5
        above before returning.
  \li 35. The user should follow the instructions for 3 <b>and</b> 5
        above before returning.
  \li 235. The user should follow the instructions for 2, 3 <b>and</b> 5
        above before returning.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if  objective function/gradient/Hessian values can be provided
    (see above)

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value of the
    objective function.

 @param[in,out] g is a one-dimensional array of size n and type rpc_, that
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] u is a one-dimensional array of size n and type rpc_, that is
    used for reverse communication (see above for details)

 @param[in,out] v is a one-dimensional array of size n and type rpc_, that is
    used for reverse communication (see above for details)
*/

//  *-*-*-   B G O _ S O L V E _ R E V E R S E _ W I T H O U T _ M A T   -*-*-*

void bgo_solve_reverse_without_mat( void **data,
                                  ipc_ *status,
                                  ipc_ *eval_status,
                                  ipc_ n,
                                  rpc_ x[],
                                  rpc_ f,
                                  rpc_ g[],
                                  rpc_ u[],
                                  rpc_ v[],
                                  ipc_ index_nz_v[],
                                  ipc_ *nnz_v,
                                  const ipc_ index_nz_u[],
                                  ipc_ nnz_u );

/*!<
 Find an approximation to the global minimizer of a given function subject to
 simple bounds on the variables using a multistart trust-region method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is
 provided by Hessian-vector products, but function/derivative information
 is only available by returning to the calling procedure.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param status (continued)
  \li  2. The user should compute the objective function value \f$f(x)\f$ at
         the point \f$x\f$ indicated in x and then re-enter the function.
         The required value should be set in f, and eval_status should be
         set to 0. If the user is unable to evaluate \f$f(x)\f$ --- for
         instance, if the function is undefined at \f$x\f$ --- the user need
         not set f, but should then set eval_status to a non-zero value.
  \li  3. The user should compute the gradient of the objective function
        \f$\nabla_x f(x)\f$ at the point \f$x\f$ indicated in x and then
        re-enter the function. The value of the i-th component of the g
        radient should be set in g[i], for i = 0, ..., n-1 and eval_status
        should be set to 0. If the user is unable to evaluate a component
        of \f$\nabla_x f(x)\f$ --- for instance if a component of the gradient
        is undefined at \f$x\f$ -the user need not set g, but should then set
        eval_status to a non-zero value.
  \li  5. The user should compute the product \f$\nabla_{xx}f(x)v\f$ of the
        Hessian of the objective function \f$\nabla_{xx}f(x)\f$ at the point
        \f$x\f$ indicated in x with the vector \f$v\f$, add the result to
        the vector \f$u\f$ and then re-enter the function. The vectors
        \f$u\f$ and \f$v\f$ are given in u and v respectively, the resulting
        vector \f$u + \nabla_{xx}f(x)v\f$ should be set
        in u and eval_status should be set to 0. If the user is unable to
        evaluate the product--- for instance, if a component of the Hessian is
        undefined at \f$x\f$ --- the user need not alter u, but should then set
        eval_status to a non-zero value.
  \li  6. The user should compute the product \f$u = P(x)v\f$ of their
        preconditioner \f$P(x)\f$ at the point x indicated in \f$x\f$ with
        the vector \f$v\f$ and then re-enter the function. The vector \f$v\f$
        is given in v, the resulting vector \f$u = P(x)v\f$ should be set in
        u and eval_status should be set to 0. If the user is unable to
        evaluate the product--- for instance, if a component of the
        preconditioner is undefined at \f$x\f$ --- the user need not set u,
        but should then set eval_status to a non-zero value.
  \li  7. The user should compute the product \f$u = \nabla_{xx}f(x)v\f$
        of the Hessian of the objective function \f$\nabla_{xx}f(x)\f$
        at the point \f$x\f$ indicated in
        x with the \b sparse vector v=\f$v\f$ and then re-enter the function.
        The nonzeros of \f$v\f$ are stored in
          v[index_nz_v[0:nnz_v-1]]
        while the nonzeros of \f$u\f$ should be returned in
          u[index_nz_u[0:nnz_u-1]];
        the user must set nnz_u and index_nz_u accordingly, and set
        eval_status to 0. If the user is unable to evaluate the product--- for
        instance, if a component of the Hessian is undefined at
        \f$x\f$--- the user
        need not alter u, but should then set eval_status to a non-zero value.
  \li 23. The user should follow the instructions for 2 <b>and</b> 3
        above before returning.
  \li 25. The user should follow the instructions for 2 <b>and</b> 5
        above before returning.
  \li 35. The user should follow the instructions for 3 <b>and</b> 5
        above before returning.
  \li 235. The user should follow the instructions for 2, 3 <b>and</b> 5
        above before returning.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if  objective function/gradient/Hessian values can be provided
    (see above)

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in]
   f is a scalar variable pointer of type rpc_, that holds the value of the
    objective function.

 @param[in,out] g is a one-dimensional array of size n and type rpc_, that
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in,out] u is a one-dimensional array of size n and type rpc_, that is
    used for reverse communication (see status=5,6,7 above for details)

 @param[in,out] v is a one-dimensional array of size n and type rpc_, that is
    used for reverse communication (see status=5,6,7 above for details)

 @param[in,out] index_nz_v is a one-dimensional array of size n and type ipc_,
    that is used for reverse communication (see status=7 above for details)

 @param[in,out] nnz_v is a scalar variable of type ipc_, that is used for
    reverse communication (see status=7 above for details)

 @param[in] index_nz_u s a one-dimensional array of size n and type ipc_,
     that is used for reverse communication (see status=7 above for details)

 @param[in] nnz_u is a scalar variable of type ipc_, that is used for reverse
     communication (see status=7 above for details). On initial (status=1)
     entry, nnz_u should be set to an (arbitrary) nonzero value, and
     nnz_u=0 is recommended
*/

//  *-*-*-*-*-*-*-*-*-*-   B G O _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void bgo_information( void **data,
                      struct bgo_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
    (see bgo_inform_type)

  @param[out] status   is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

//  *-*-*-*-*-*-*-*-*-*-   B G O _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void bgo_terminate( void **data,
                    struct bgo_control_type *control,
                    struct bgo_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see bgo_control_type)

  @param[out] inform   is a struct containing output information
              (see bgo_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example bgot.c
   This is an example of how to use the package to find an approximation
   to the global minimum of a given function within a bounded region.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example bgotf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
