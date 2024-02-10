//* \file galahad_bqp.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BQP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. February 21st 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package bqp

  \section bqp_intro Introduction

  \subsection bqp_purpose Purpose

  This package uses  a preconditioned, projected-gradient method
  to solve the <b>convex bound-constrained quadratic programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  subject to the simple bound constraints
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the \f$n\f$ by \f$n\f$ symmetric postive semi-definite matrix \f$H\f$,
  the vectors \f$g\f$, \f$x^l\f$, \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$H\f$;
  the matrix need not be provided as there are options to obtain matrix-vector
  products involving \f$H\f$ by reverse communication.

  \subsection bqp_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection bqp_date Originally released

  November 2009, C interface February 2022.

  \subsection bqp_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[x^l \leq x \leq x^u,\f]
\manonly
  \n
   x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[H x + g = z\f]
\manonly
  \n
   H x + g = z
  \n
\endmanonly
  where
  \f[ z = z^l + z^u, \,\,
   z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,\f]
\manonly
  \n
   z = z^l + z^u, z^l \[>=] 0 and z^u \[<=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[(x -x^l )^{T} z^l = 0 \;\;  \mbox{and} \;\; (x -x^u )^{T} z^u = 0,\hspace{12mm} \f]
\manonly
  \n
  (x -x^l)^T z^l = 0 and (x -x^u)^T z^u = 0,
  \n
\endmanonly
  where the vector \f$z\f$ is known as  the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection bqp_method Method

  The method is iterative. Each iteration proceeds in two stages.
  Firstly, the so-called generalized Cauchy point for the quadratic
  objective is found.  (The purpose of this point is to ensure that the
  algorithm converges and that the set of bounds which are satisfied as
  equations at the solution is rapidly identified.)  Thereafter an
  improvement to the objective is sought using either a
  direct-matrix or truncated conjugate-gradient algorithm.

  \subsection bqp_references Reference

  This is a specialised version of the method presented in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint (1988).
  Global convergence of a class of trust region algorithms
  for optimization with simple bounds.
  SIAM Journal on Numerical Analysis <b>25</b> 433-460,

  \subsection bqp_call_order Call order

  To solve a given problem, functions from the bqp package must be called
  in the following order:

  - \link bqp_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link bqp_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - set up problem data structures and fixed values by caling one of
     - \link bqp_import \endlink - in the case that \f$H\f$ is explicitly
        available
     - \link bqp_import_without_h \endlink - in the case that only the
        effect of applying \f$H\f$ to a vector is possible
  - \link bqp_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link bqp_solve_given_h \endlink - solve the problem using values
          of \f$H\f$
     - \link bqp_solve_reverse_h_prod \endlink - solve the problem by returning
         to the caller for products of \f$H\f$ with specified vectors
  - \link bqp_information \endlink (optional) - recover information about
    the solution and solution process
  - \link bqp_terminate \endlink - deallocate data structures

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

  If it is explicitly available, the symmetric \f$n\f$ by \f$n\f$
  objective Hessian matrix \f$H\f$ may be presented
  and stored in a variety of formats. But crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

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
#ifndef GALAHAD_BQP_H
#define GALAHAD_BQP_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sbls.h"

/**
 * control derived type as a C struct
 */
struct bqp_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit number for error and warning diagnostics
    ipc_ error;

    /// \brief
    /// general output unit number
    ipc_ out;

    /// \brief
    /// the level of output required
    ipc_ print_level;

    /// \brief
    /// on which iteration to start printing
    ipc_ start_print;

    /// \brief
    /// on which iteration to stop printing
    ipc_ stop_print;

    /// \brief
    /// how many iterations between printing
    ipc_ print_gap;

    /// \brief
    /// how many iterations to perform (-ve reverts to HUGE(1)-1)
    ipc_ maxit;

    /// \brief
    /// cold_start should be set to 0 if a warm start is required (with variable
    /// assigned according to B_stat, see below), and to any other value if the
    /// values given in prob.X suffice
    ipc_ cold_start;

    /// \brief
    /// the ratio of how many iterations use CG rather steepest descent
    ipc_ ratio_cg_vs_sd;

    /// \brief
    /// the maximum number of per-iteration changes in the working set permitted
    /// when allowing CG rather than steepest descent
    ipc_ change_max;

    /// \brief
    /// how many CG iterations to perform per BQP iteration (-ve reverts to n+1)
    ipc_ cg_maxit;

    /// \brief
    /// the unit number to write generated SIF file describing the current
    /// problem
    ipc_ sif_file_device;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// the required accuracy for the primal infeasibility
    rpc_ stop_p;

    /// \brief
    /// the required accuracy for the dual infeasibility
    rpc_ stop_d;

    /// \brief
    /// the required accuracy for the complementary slackness
    rpc_ stop_c;

    /// \brief
    /// any pair of constraint bounds (x_l,x_u) that are closer than i
    /// dentical_bounds_tol will be reset to the average of their values
    ///
    rpc_ identical_bounds_tol;

    /// \brief
    /// the CG iteration will be stopped as soon as the current norm of the
    /// preconditioned gradient is smaller than
    /// max( stop_cg_relative * initial preconditioned gradient,
    /// stop_cg_absolute)
    rpc_ stop_cg_relative;
    /// see stop_cg_relative
    rpc_ stop_cg_absolute;

    /// \brief
    /// threshold below which curvature is regarded as zero
    rpc_ zero_curvature;

    /// \brief
    /// the maximum CPU time allowed (-ve = no limit)
    rpc_ cpu_time_limit;

    /// \brief
    /// exact_arcsearch is true if an exact arcsearch is required, and false if
    /// approximation suffices
    bool exact_arcsearch;

    /// \brief
    /// if space_critical is true, every effort will be made to use as little
    /// space as possible. This may result in longer computation times
    bool space_critical;

    /// \brief
    /// if deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if generate_sif_file is true, a SIF file describing the current problem
    /// will be generated
    bool generate_sif_file;

    /// \brief
    /// name (max 30 characters) of generated SIF file containing input problem
    char sif_file_name[31];

    /// \brief
    /// all output lines will be prefixed by a string (max 30 characters)
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    ///
    char prefix[31];

    /// \brief
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;
};

/**
 * time derived type as a C struct
 */
struct bqp_time_type {

    /// \brief
    /// total time
    real_sp_ total;

    /// \brief
    /// time for the analysis phase
    real_sp_ analyse;

    /// \brief
    /// time for the factorization phase
    real_sp_ factorize;

    /// \brief
    /// time for the linear solution phase
    real_sp_ solve;
};

/**
 * inform derived type as a C struct
 */
struct bqp_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0  success
    /// \li -1  allocation error
    /// \li -2  deallocation error
    /// \li -3  matrix data faulty (.n < 1, .ne < 0)
    /// \li -20  alegedly +ve definite matrix is not
    ipc_ status;

    /// \brief
    /// Fortran STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// status return from factorization
    ipc_ factorization_status;

    /// \brief
    /// number of iterations required
    ipc_ iter;

    /// \brief
    /// number of CG iterations required
    ipc_ cg_iter;

    /// \brief
    /// current value of the objective function
    rpc_ obj;

    /// \brief
    /// current value of the projected gradient
    rpc_ norm_pg;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// times for various stages
    struct bqp_time_type time;

    /// \brief
    /// inform values from SBLS
    struct sbls_inform_type sbls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    B Q P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void bqp_initialize( void **data,
                     struct bqp_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see bqp_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    B Q P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void bqp_read_specfile( struct bqp_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNBQP.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/bqp.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see bqp_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    B Q P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void bqp_import( struct bqp_control_type *control,
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
  paramters for the remaining prcedures (see bqp_control_type)

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
    variables.

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

// *-*-*-*-*-*-*-*-    B Q P  _ I M P O R T _ W I T H O U T _ H   -*-*-*-*-*-*

void bqp_import_without_h( struct bqp_control_type *control,
                           void **data,
                           ipc_ *status,
                           ipc_ n );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bqp_control_type)

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
  \li -3. The restriction n > 0 has been violated.

@param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

*/

// *-*-*-*-*-*-*-    B Q P  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void bqp_reset_control( struct bqp_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bqp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-   B Q P _ S O L V E _ G I V E N _ H   -*-*-*-*-*-*-*-*-

void bqp_solve_given_h( void **data,
                        ipc_ *status,
                        ipc_ n,
                        ipc_ h_ne,
                        const rpc_ H_val[],
                        const rpc_ g[],
                        const rpc_ f,
                        const rpc_ x_l[],
                        const rpc_ x_u[],
                        rpc_ x[],
                        rpc_ z[],
                        ipc_ x_stat[] );

/*!<
 Solve the bound-constrained quadratic program when the Hessian \f$H\f$
is available.

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
  \li -3. The restriction n > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows' or
       'diagonal' has been violated.
  \li -4. The simple-bound constraints are inconsistent.
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
  \li -17. The step is too small to make further impact.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -20. The Hessian matrix \f$H\f$ appears to be indefinite.
           specified.
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] h_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] x_l is a one-dimensional array of size n and type rpc_, that
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_, that
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.
*/

//  *-*-*-*-*-*-   B Q P _ S O L V E _ R E V E R S E _ H + P R O D   -*-*-*-*-*-

void bqp_solve_reverse_h_prod( void **data,
                                ipc_ *status,
                                ipc_ n,
                                const rpc_ g[],
                                const rpc_ f,
                                const rpc_ x_l[],
                                const rpc_ x_u[],
                                rpc_ x[],
                                rpc_ z[],
                                ipc_ x_stat[],
                                rpc_ v[],
                                const rpc_ prod[],
                                ipc_ nz_v[],
                                ipc_ *nz_v_start,
                                ipc_ *nz_v_end,
                                const ipc_ nz_prod[],
                                ipc_ nz_prod_end );

/*!<
 Solve the bound-constrained quadratic program when the products of the
Hessian \f$H\f$ with specified vectors may be computed by the calling
program.

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
  \li -3. The restriction n > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows' or
       'diagonal' has been violated.
  \li -4. The simple-bound constraints are inconsistent.
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
  \li -17. The step is too small to make further impact.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -20. The Hessian matrix \f$H\f$ appears to be indefinite.
           specified.
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
           specified.

 @param status (continued)
  \li  2. The product \f$Hv\f$ of the Hessian \f$H\f$ with a given output
       vector \f$v\f$ is required from the user. The vector \f$v\f$ will be
       stored in v and the product \f$Hv\f$ must be returned in prod, and
       bqp_solve_reverse_h_prod re-entered with all other arguments unchanged.
  \li 3. The product \f$Hv\f$ of the Hessian H with a given output vector
       \f$v\f$ is required from the user. Only components
        nz_v[nz_v_start-1:nz_v_end-1]
       of the vector \f$v\f$ stored in v are nonzero. The resulting
       product \f$Hv\f$ must be placed in prod, and bqp_solve_reverse_h_prod
       re-entered with all other arguments unchanged.
  \li 4. The product \f$Hv\f$ of the Hessian H with a given output vector
       \f$v\f$ is required from the user. Only components
        nz_v[nz_v_start-1:nz_v_end-1]
       of the vector \f$v\f$  stored in v are nonzero. The resulting
       <b>nonzeros</b> in the product \f$Hv\f$ must be placed in their
       appropriate comnponents of prod, while a list of indices of the nonzeros
       placed in
         nz_prod[0 : nz_prod_end-1].
       bqp_solve_reverse_h_prod should then be re-entered with all other
       arguments unchanged. Typically v will be very sparse
       (i.e., nz_p_end-nz_p_start will be small).

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] x_l is a one-dimensional array of size n and type rpc_, that
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_, that
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] v is a one-dimensional array of size n and type rpc_, that
    is used for reverse communication (see status=2-4 above for details)

 @param[in] prod is a one-dimensional array of size n and type rpc_, that
    is used for reverse communication (see status=2-4 above for details)

 @param[out] nz_v is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_start is a scalar of type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_end is a scalar of type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[in] nz_prod is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=4 above for details)

 @param[in] nz_prod_end is a scalar of type ipc_, that
    is used for reverse communication (see status=4 above for details)

*/

// *-*-*-*-*-*-*-*-*-*-    B Q P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void bqp_information( void **data,
                      struct bqp_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see bqp_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    B Q P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void bqp_terminate( void **data,
                    struct bqp_control_type *control,
                    struct bqp_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see bqp_control_type)

  @param[out] inform   is a struct containing output information
              (see bqp_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example bqpt.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example bqptf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
