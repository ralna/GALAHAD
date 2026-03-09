//* \file galahad_snls.h */

/*
 * THIS VERSION: GALAHAD 5.5 - 2026-03-08 AT 09:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SNLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. May 26th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package snls

  \section snls_intro Introduction

  \subsection snls_purpose Purpose

  This package uses a regularized Gauss-Newton method to solve the
   <b>simplex-constrained nonlinear least-squares problem</b>
  \f[\mbox{minimize}\;\; f(x) = \frac{1}{2} \sum_{i=1}^{m_r} w_i^{} r_i^2(x) 
 \equiv  \frac{1}{2} \| r(x) \|_W^2 \f],
\manonly
  \n
  minimize f(x) = 1/2 sum_{i=1}^m w_i c_i^2(x) = 1/2 || r(x) ||_W^2,
  \n
\endmanonly
  where \f$x\f$ is required to lie in the regular simplex
  \f[e^T x = 1 \;\;\mbox{and}\;\;  x_j \geq 0, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   e^T x = 1 and x_j \[>=] 0, j = 1, ... , n,
  \n
\endmanonly
  where the <b>residual</b> function \f$r(x)f$ maps from \f$R^n\f$ to
 \f$R^o\f$, and the \f$R^o\f$ non-negative diagonal weights\f$\w\f$ are given.
  Full advantage is taken of any zero coefficients of the 
  <b>Jacobian matrix</b> \f$J_r(x) \equiv \nabla r(x)f$;  the matrix need not 
  be provided explicitly as there are options to obtain matrix-vector products 
  involving \f$J_r(x)\f$ and its transpose either
  by reverse communication or from a user-provided subroutine.
  \subsection snls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection snls_date Originally released

  February 2026.

  \subsection snls_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[e^T x = 1 \;\;\mbox{and}\;\; x \geq 0 ,\f]
\manonly
  \n
   e^T x = 1 and x [>=] 0,
  \n
\endmanonly
  the dual optimality conditions
  \f[(J_r^T(x) r(x) = e y + z\f]
\manonly
  \n
   J_r^T(x)r(x) = e y + z
  \n
\endmanonly
  where the dual variables
  \f[ z \geq 0,\f]
\manonly
  \n
   zl \[>=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[x^T z = 0,\hspace{12mm} \f]
\manonly
  \n
  x^T z = 0,
  \n
\endmanonly
  where the vector inequalities hold component-wise.

  \subsection snls_method Method

  Blah, blah, blah (see pdf docs)

  \subsection snls_references Reference

  Full details are provided in

  N. I. M. Gould (2026).
  Nonlinear least-squares over unit simplices.
  In preparation.

  \subsection snls_call_order Call order

  To solve a given problem, functions from the snls package must be called
  in the following order:

  - \link snls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link snls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - set up problem data structures and fixed values by caling one of
     - \link snls_import \endlink - in the case that \f$J_r(x)\f$ is explicitly
        available
     - \link snls_import_without_jac \endlink - in the case that only the
        effect of applying \f$J_r\f$ and its transpose to a vector is possible
  - \link snls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link snls_solve_with_jac \endlink - solve using function calls to
       evaluate residual and Jacobian values
     - \link snls_solve_with_jacprod \endlink - solve using function calls to
       evaluate residual values and Jacobian-vector products
     - \link snls_solve_reverse_with_jac \endlink - solve returning to the
       calling program to obtain residual and Jacobian values, or
     - \link snls_solve_reverse_with_jacprod \endlink - solve returning to the
       calling prorgram to obtain residual values and Jacobian-vector products
  - \link snls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link snls_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$m\f$ by \f$n\f$ matrix \f$J_r\f$ may be presented
  and stored in a variety of convenient input formats.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense_row Dense row storage format

  The matrix \f$J_r\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array Jr_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_dense_column Dense column storage format

  The matrix \f$J_r\f$ is stored as a compact  dense matrix by columns, that is,
  the values of the entries of each column in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$m \ast j + i\f$  of the storage array Jr_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$J_r\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays Jr_row and
  Jr_col and real array Jr_val, respectively, while the number of nonzeros
  is recorded as Jr_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$J_r\f$ the i-th component of the
  integer array Jr_ptr holds the position of the first entry in this row,
  while Jr_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = Jr_ptr(i), \f$\ldots\f$, Jr_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array Jr_col, and real array Jr_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessors.

  \subsubsection unsymmetric_matrix_column_wise Sparse column-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in column j appear directly before those
  in column j+1. For the j-th column of \f$J_r\f$ the j-th component of the
  integer array Jr_ptr holds the position of the first entry in this column,
  while Jr_ptr(n) holds the total number of entries.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values \f$A_{ij}\f$
  of the  nonzero entries in the j-th column are stored in components
  l = Jr_ptr(j), \f$\ldots\f$, Jr_ptr(j+1)-1,  \f$0 \leq j \leq n-1\f$,
  of the integer array Jr_row, and real array Jr_val, respectively.
  Once again, for sparse matrices, this scheme almost always requires less
  storage than the dense of coordinate formats.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SNLS_H
#define GALAHAD_SNLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// callbacks
#include "galahad_callbacks.h"

// required packages
#include "galahad_slls.h"
#include "galahad_sllsb.h"

/**
 * control derived type as a C struct
 */
struct snls_control_type {

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
    /// removal of the file alive_file from unit alive_unit terminates execution
    ipc_ alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// is the Jacobian matrix of first derivatives available (\f$\geq\f$ 2),
    /// is access only via matrix-vector products (=1) or is it not
    /// available (\f$\leq\f$ 0) ?
    ipc_ jacobian_available;

    /// \brief
    ///  specify the method to be used to solve the subproblem
    /// 1  use a projection method (slls)
    /// 2  use an interior-point method (sllsb)
    /// 3  start with an interior-point method but later switch to projection
    ipc_ subproblem_solver;

    /// \brief
    /// non-monotone \f$\leq\f$ 0 monotone strategy used, anything else
    /// non-monotone strategy with this history length used
    ipc_ non_monotone;

    /// \brief
    /// define the weight-update strategy:
    /// 1 (basic), 2 (reset to zero when very successful),
    /// 3 (imitate TR), 4 (increase lower bound), 5 (GPT)
    ipc_ weight_update_strategy;

    /// \brief
    /// overall convergence tolerances. The iteration will terminate when
    /// \f$||c(x)||_2 \leq \f$ MAX( .stop_c_absolute, .stop_c_relative
    ///   \f$ * \|c(x_{\mbox{initial}})\|_2\f$ or
    /// when the norm of the projected gradient, 
    /// \f$pg = p[x-J_r^T(x) r(x)]-x\|_2\f$, satisfies
    /// \f$\|pg\|_2 \leq\f$  MAX( .stop_pg_absolute, .stop_pg_relative
    ///   \f$ * \|pg_{\mbox{initial}}\|_2\f$, or
    /// if the step is less than .stop_s
    rpc_ stop_r_absolute;
    /// see stop_r_absolute
    rpc_ stop_r_relative;
    /// see stop_r_absolute
    rpc_ stop_pg_absolute;
    /// see stop_r_absolute
    rpc_ stop_pg_relative;
    /// see stop_r_absolute
    rpc_ stop_s;

    /// \brief
    ///  The iteration will switch from an interior-point to a projection 
    /// solver when .subproblem_solver = 3 if ||pg||_2 satisfies ||pg||_2 <= 
    /// \f$\|pg\|_2 \leq\f$  MAX( .stop_pg_absolute, .stop_pg_switch
    ///   \f$ * \|pg_{\mbox{initial}}\|_2\f$
    rpc_ stop_pg_switch;


    /// \brief
    /// initial value for the regularization weight (-ve => \f$1/\|g_0\|)\f$)
    rpc_ initial_weight;

    /// \brief
    /// minimum permitted regularization weight
    rpc_ minimum_weight;

    /// \brief
    /// a potential iterate will only be accepted if the actual decrease
    /// f - f(x_new) is larger than .eta_successful times that predicted
    /// by a quadratic model of the decrease. The regularization weight will be
    /// decreaed if this relative decrease is greater than .eta_very_successful
    /// but smaller than .eta_too_successful
    rpc_ eta_successful;
    /// see eta_successful
    rpc_ eta_very_successful;
    /// see eta_successful
    rpc_ eta_too_successful;

    /// \brief
    /// on very successful iterations, the regularization weight will be reduced
    /// by the factor .weight_decrease but no more than .weight_decrease_min
    /// while if the iteration is unsucceful, the weight will be increased by a
    /// factor .weight_increase but no more than .weight_increase_max
    /// (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
    /// and Toint, 2011)
    rpc_ weight_decrease_min;
    /// see weight_decrease_min
    rpc_ weight_decrease;
    /// see weight_decrease_min
    rpc_ weight_increase;
    /// see weight_decrease_min
    rpc_ weight_increase_max;

    /// \brief
    /// the value of the two-norm of the projected gradient required before
    /// a switch is made from the Gauss-Newton to the Newton model when 
    /// .newton_acceleration is true
    rpc_ switch_to_newton;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// should second derivatives be used to accelerate the convergence of the 
    /// algorithm? (Not yet implemented)
    bool newton_acceleration;

    /// \brief
    /// allow the user to perform a "magic" step to improve the objective
    bool magic_step;

    /// \brief
    /// print values of the objective/projected gradient rather than ||r|| 
    /// and  the projected gradient
    bool print_obj;

    /// \brief
    /// if space_critical is true, every effort will be made to use as little
    /// space as possible. This may result in longer computation times
    bool space_critical;

    /// \brief
    /// if deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by a string (max 30 characters)
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    ///
    char prefix[31];

    /// \brief
    /// control parameters for SLLS
    struct slls_control_type slls_control;

    /// \brief
    /// control parameters for SLLSB
    struct sllsb_control_type sllsb_control;
};

/**
 * time derived type as a C struct
 */
struct snls_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the total CPU time spent in the slls package
    rpc_ slls;

    /// \brief
    /// the total CPU time spent in the sllsb package
    rpc_ sllsb;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the total clock time spent in the slls package
    rpc_ clock_slls;

    /// \brief
    /// the total clock time spent in the sllsb package
    rpc_ clock_sllsb;

};

/**
 * inform derived type as a C struct
 */
struct snls_inform_type {

    /// \brief
    /// reported return status.
    ipc_ status;

    /// \brief
    /// Fortran STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the name of the user-supplied evaluation routine for which an error
    /// occurred
    char bad_eval[13];

    /// \brief
    /// the total number of iterations performed
    ipc_ iter;

    /// \brief
    /// the total number of inner iterations performed
    ipc_ inner_iter;

    /// \brief
    /// the total number of evaluations of the residual function r(x)
    ipc_ r_eval;

    /// \brief
    /// the total number of evaluations of the Jacobian J_r(x) of r(x)
    ipc_ jr_eval;

    /// \brief
    /// the value of the objective function \f$\frac{1}{2}\|r(x)\|^2_W\f$
    /// at the best estimate the solution, x, determined by snls_solve
    rpc_ obj;

    /// \brief
    /// the norm of the residual \f$\|r(x)\|_W\f$ at the best estimate of
    /// the solution x, determined by snls_solve
    rpc_ norm_r;

    /// \brief
    /// the norm of the gradient of \f$\|r(x)\|_W\f$ of the objective function
    /// at the best estimate, x, of the solution determined by snls_solve
    rpc_ norm_g;

    /// \brief
    /// the norm of the projected gradient of \f$\|r(x)\|_W\f$ at the best 
    /// estimate, x, of the solution determined by snls_solve
    rpc_ norm_pg;

    /// \brief
    /// the final regularization weight used
    rpc_ weight;

    /// \brief
    /// times for various stages
    struct snls_time_type time;

    /// \brief
    /// inform values from slls
    struct slls_inform_type slls_inform;

    /// \brief
    /// inform values for sllsb
    struct sllsb_inform_type sllsb_inform;

    /// \brief
    /// the output flag from LAPACK routines
    ipc_ lapack_error;

};

// *-*-*-*-*-*-*-*-*-*-    S N L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void snls_initialize( void **data,
                      struct snls_control_type *control,
                      struct snls_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see snls_control_type)

  @param[out] inform  is a struct containing output information
              (see snls_inform_type)

*/

// *-*-*-*-*-*-*-*-*-    S N L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void snls_read_specfile( struct snls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNSNLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/snls.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see snls_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    S N L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void snls_import( struct snls_control_type *control,
                  void **data,
                  ipc_ *status,
                  ipc_ n,
                  ipc_ m_r,
                  ipc_ m_c,
                  const char Jr_type[],
                  ipc_ Jr_ne,
                  const ipc_ Jr_row[],
                  const ipc_ Jr_col[],
                  ipc_ Jr_ptr_ne,
                  const ipc_ Jr_ptr[],
                  const ipc_ cohort[]);

/*!<
 Import problem data including the structure of \f$J_r(x)\f$ 
 into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see snls_control_type)

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
  \li -3. The restrictions n > 0, m_r > 0 or requirement that type contains
       its relevant string 'coordinate', 'sparse_by_rows',
       'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns';
       has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of
   cohorts.

 @param[in]  Jr_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the objective design matrix, \f$J_r\f$. It should be one of
   'coordinate', 'sparse_by_rows', 'sparse_by_columns',
   'dense' or 'dense_by_columns'; lower or upper case variants are allowed.

 @param[in]  Jr_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$J_r\f$ in the sparse storage schemes.
   It need not be set for either of the dense schemes.

 @param[in]  Jr_row is a one-dimensional array of size Jr_ne and type ipc_, that
   holds the row indices of \f$J_r\f$ in the sparse co-ordinate
   or sparse column-wise storage schemes. It need not be set for any of the
   other schemes, and in this case can be NULL.

 @param[in]  Jr_col is a one-dimensional array of size Jr_ne and type ipc_,
   that holds the column indices of \f$J_r\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  Jr_ptr_ne is a scalar variable of type ipc_, that holds the
   length of the pointer array if sparse row or column storage scheme is
   used for \f$J_r\f$. For the sparse row scheme,  Jr_ptr_ne should be at least
   o+1, while for the sparse column scheme,  it should be at least n+1,
   It need not be set when the other schemes are used.

 @param[in]  Jr_ptr is a one-dimensional array of size o+1 and type ipc_,
   that holds the starting position of each row of \f$J_r\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   By contrast, it is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each column of \f$J_r\f$, as well as the
   total number of entries, in the sparse column-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.

 @param[in]  cohort is a one-dimensional array of size n and type ipc_,
    that specifies which cohort each variable is assigned to.
    If variable $x_j$ is associated with cohort $\cal C_i$, 
    $0 \leq i \leq m-1$, cohort[j] should be set to i, while 
    if $x_j$ is unconstrained cohort[j] = 0 should be assigned. 
    At least one value cohort[j] for $j = 0,\ldots\,n-1$ is expected 
    to take the value $i$ for every $0 \leq i \leq m-1$, that is 
    no empty cohorts are allowed. If all the variables lie in a
    single simplex, cohort can be set to NULL.

*/

// *-*-*-*-*-*-*-    S N L S  _ I M P O R T _ W I T H O U T _ A   -*-*-*-*-*-*

void snls_import_without_jac( struct snls_control_type *control,
                              void **data,
                              ipc_ *status,
                              ipc_ n,
                              ipc_ m_r,
                              ipc_ m_c,
                              const ipc_ cohort[]);

/*!<
 Import problem data excluding the structure of \f$J_r(x)\f$,
 into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see snls_control_type)

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
  \li -3. The restriction n > 0 or m_r > 0 has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of
   cohorts.

 @param[in]  cohort is a one-dimensional array of size n and type ipc_,
    that specifies which cohort each variable is assigned to.
    If variable $x_j$ is associated with cohort $\cal C_i$, 
    $0 \leq i \leq m-1$, cohort[j] should be set to i, while 
    if $x_j$ is unconstrained cohort[j] = 0 should be assigned. 
    At least one value cohort[j] for $j = 0,\ldots\,n-1$ is expected 
    to take the value $i$ for every $0 \leq i \leq m-1$, that is 
    no empty cohorts are allowed. If all the variables lie in a
    single simplex, cohort can be set to NULL.

*/
// *-*-*-*-*-*-*-    S N L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void snls_reset_control( struct snls_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see snls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/
//  *-*-*-*-*-*-*-*-*-  S N L S _ S O L V E _ W I T H _ J A C   -*-*-*-*-*-*-*-*

void snls_solve_with_jac( void **data,
                          void *userdata,
                          ipc_ *status,
                          ipc_ n,
                          ipc_ m_r,
                          ipc_ m_c,
                          rpc_ x[],
                          rpc_ y[],
                          rpc_ z[],
                          rpc_ r[],
                          rpc_ g[],
                          ipc_ x_stat[],
                          galahad_r *eval_r,
                          ipc_ jr_ne,
                          galahad_jr *eval_jr,
                          const rpc_ w[]);

/*!<
 Find a local minimizer of a given function using a trust-region method.

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
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of 
   cohorts.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers.
    The i-th component of y, i = 0, ... , o-1, contains \f$y_i\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] r is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the residuals \f$r(x)\f$.
    The i-th component of r, i = 0, ... , o-1, contains \f$r_i(x)\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g(x) = J_r(x)^T r(x)\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j(x)\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param eval_r is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_r( ipc_ n, ipc_ m, const rpc_ x[], rpc_ r[],
                   const void *userdata )
   \endcode
   The componnts of the residual function \f$r(x)\f$ evaluated at x=\f$x\f$
   must be assigned to r, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_r via the structure \c userdata.

 @param[in] jr_ne is a scalar variable of type ipc_, that holds the number of
    entries in the Jacobian matrix \f$J_r(x)\f$.

 @param eval_jr is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_jr( ipc_ n, ipc_ m_r, ipc_ jr_ne, const rpc_ x[], rpc_ jr_val[],
                    const void *userdata )
   \endcode
   The components of the Jacobian \f$J_r(x) = \nabla_x r(x\f$) of the residuals
   must be assigned to jr_val in the same order as presented to snls_import, and
   the function return value set to 0. If the evaluation is impossible at x,
   return should be set to a nonzero value.
   Data may be passed into \c eval_jr via the structure \c userdata.

 @param[in] w is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the weights \f$w\f$. The j-th component of w, 
    j = 0, ... , n-1, contains \f$w_j\f$. It need not be set if the weights 
    are all ones, and in this case can be NULL.

 */

//  -*-*-*-*-*-*-   S N L S _ S O L V E _ W I T H _ J A C P R O D   -*-*-*-*-*-

void snls_solve_with_jacprod( void **data,
                              void *userdata,
                              ipc_ *status,
                              ipc_ n,
                              ipc_ m_r,
                              ipc_ m_c,
                              rpc_ x[],
                              rpc_ y[],
                              rpc_ z[],
                              rpc_ r[],
                              rpc_ g[],
                              ipc_ x_stat[],
                              galahad_r *eval_r,
                              galahad_jr_prod *eval_jr_prod,
                              galahad_jr_scol *eval_jr_scol,
                              galahad_jr_sprod *eval_jr_sprod,
                              const rpc_ w[]);

/*!<
 Solve the simplex-constrained nonlinear least-squares problem when the
 products of the Jacobian \f$J_r(x)\f$ and its transpose are available 
 by function calls.

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
  \li -82. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of 
   cohorts.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers.
    The i-th component of y, i = 0, ... , o-1, contains \f$y_i\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] r is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the residuals \f$r(x)\f$.
    The i-th component of r, i = 0, ... , o-1, contains \f$r_i(x)\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g(x) = J_r(x)^T r(x)\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j(x)\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param eval_r is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_r( ipc_ n, const rpc_ x[], rpc_ c[], const void *userdata )
   \endcode
   The componnts of the residual function \f$c(x)\f$ evaluated at x=\f$x\f$
   must be assigned to c, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_r via the structure \c userdata.

 @param eval_jr_prod is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_jr_prod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
                         const rpc_ v[], rpc_ p[], bool got_jr,
                         const void *userdata )
   \endcode
   The product\f$J_r(x) v\f$ (if tranpose is false) or \f$J_r^T(x) v\f$ 
   (if tranpose is true) bewteen the Jacobian 
   \f$J_r(x) = \nabla_{x}r(x)\f$ or its tranpose with the vector v=\f$v\f$
   must be returned in p, and the function return value set to 0. If 
   the evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_jr_prod via the structure \c userdata.

 @param eval_jr_scol is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_jr_scol( ipc_ n, ipc_ m_r, const rpc_ x[], ipc_ index,
                         rpc_ val[], ipc_ row[], ipc_ nz, bool got_jr,
                         const void *userdata )
   \endcode
   The nonzeros and corresponding row entries of the index-th colum of 
   \f$J_r(x)\f$ must be returned in val and row, respectively, together
   with the number of entries, nz, and the function return value set to 0. 
   If the evaluation is impossible at x, return should be set to a nonzero 
   value. Data may be passed into \c eval_jr_scol via the structure \c userdata.

 @param eval_jr_sprod is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_jr_sprod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
                          const rpc_ v[], rpc_ p[], const ipc_ free,
                          ipc_ n_free, bool got_jr, const void *userdata )
   \endcode
   The product\f$J_r(x) v\f$ (if tranpose is false) or \f$J_r^T(x) v\f$ 
   (if tranpose is true) bewteen the Jacobian 
   \f$J_r(x) = \nabla_{x}r(x)\f$ or its tranpose with the vector v=\f$v\f$
   must be returned in u, and the function return value set to 0. If
   transpose is false, only the components free[0 : n_free-1] of 
   $v$ will be nonzero, while if transpose is true, only the components 
   free[0 : n_free-1] of p should be set. If the evaluation is 
   impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_jr_srprod via the structure \c userdata.

 @param[in] w is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the weights \f$w\f$. The j-th component of w, 
    j = 0, ... , n-1, contains \f$w_j\f$. It need not be set if the weights 
    are all ones, and in this case can be NULL.

 */


//  -*-*-*-*-   S N L S _ S O L V E _ R E V E R S E _ W I T H  _ J A C  -*-*-*-

void snls_solve_reverse_with_jac( void **data,
                                  ipc_ *status,
                                  ipc_ *eval_status,
                                  ipc_ n,
                                  ipc_ m_r,
                                  ipc_ m_c,
                                  rpc_ x[],
                                  rpc_ y[],
                                  rpc_ z[],
                                  rpc_ r[],
                                  rpc_ g[],
                                  ipc_ x_stat[],
                                  ipc_ jr_ne,
                                  const rpc_ jr_val[],
                                  const rpc_ w[]);

/*!<
 Solve the simplex-constrained nonlinear least-squares problem when the
 products of the Jacobian \f$J_r(x)\f$ and its transpose
 with specified vectors may be computed by the calling program.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    Possible exit are:
  \li  0. The run was successful.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that a type contains
       its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns',
       'dense_by_rows' or 'dense_by_columns' has been violated.
  \li -4. The simple-bound constraints are inconsistent.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.

 @param status (continued)
  \li  2. The user should compute the vector of residuals \f$r(x)\f$ at
         the point \f$x\f$ indicated in x and then re-enter the function.
         The required value should be set in r, and eval_status should be
         set to 0. If the user is unable to evaluate \f$c(x)\f$--- for
         instance, if the function is undefined at \f$x\f$--- the user need
         not set r, but should then set eval_status to a non-zero value.

  \li  3. The user should compute the Jacobian of the vector of residual
         functions, \f$J_r(x) = \nabla_x c(x)\f$, at the point \f$x\f$ 
         indicated in x
         and then re-enter the function. The l-th component of the Jacobian
         stored according to the scheme specified for the remainder of
         \f$Jr\f$ in the earlier call to nls_import should be set in Jr_val[l],
         for l = 0, ..., J_ne-1 and eval_status should be set to 0.
         If the user is unable to evaluate a component of \f$J_r\f$ --- for
         instance, if a component of the matrix is undefined at
         \f$x\f$ --- the user need not set Jr_val, but should
         then set eval_status to a non-zero value.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if the matrix products can be provided (see \c status above)

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of 
   cohorts.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers.
    The i-th component of y, i = 0, ... , o-1, contains \f$y_i\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] r is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the residuals \f$r(x)\f$.
    The i-th component of r, i = 0, ... , o-1, contains \f$r_i(x)\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g(x) = J_r(x)^T r(x)\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j(x)\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[in] Jr_ne is a scalar variable of type ipc_, that holds the number of
    entries in the Jacobian \f$J_r(x)\f$.

 @param[in] Jr_val is a one-dimensional array of size A_ne and type rpc_,
    that holds the values of the entries of the Jacobian \f$J_r(x)\f$
    in any of the available storage schemes.

 @param[in] w is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the weights \f$w\f$. The j-th component of w, 
    j = 0, ... , n-1, contains \f$w_j\f$. It need not be set if the weights 
    are all ones, and in this case can be NULL.

*/
//  -*-   S N L S _ S O L V E _ R E V E R S E _ W I T H  _ J A C P R O D   -*-

void snls_solve_reverse_with_jacprod( void **data,
                                      ipc_ *status,
                                      ipc_ *eval_status,
                                      ipc_ n,
                                      ipc_ m_r,
                                      ipc_ m_c,
                                      rpc_ x[],
                                      rpc_ y[],
                                      rpc_ z[],
                                      rpc_ r[],
                                      rpc_ g[],
                                      ipc_ x_stat[],
                                      rpc_ v[],
                                      ipc_ iv[],
                                      ipc_ *lvl,
                                      ipc_ *lvu,
                                      ipc_ *index,
                                      const rpc_ p[],
                                      const ipc_ ip[],
                                      ipc_ lp,
                                      const rpc_ w[]);

/*!<
 Solve the simplex-constrained nonlinear least-squares problem when the
 products of the Jacobian \f$J_r(x)\f$ and its transpose
 with specified vectors may be computed by the calling program.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    Possible exit are:
  \li  0. The run was successful.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restriction n > 0 or requirement that a type contains
       its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns',
       'dense_by_rows' or 'dense_by_columns' has been violated.
  \li -4. The simple-bound constraints are inconsistent.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.

 @param status (continued)
  \li  2. The user should compute the vector of residuals \f$r(x)\f$ at
         the point \f$x\f$ indicated in x and then re-enter the function.
         The required value should be set in r, and eval_status should be
         set to 0, and snls_solve_reverse_with_jacprod re-entered with all 
         other arguments unchanged. If the user is unable to evaluate 
         \f$c(x)\f$---for instance, if the function is undefined at 
         \f$x\f$--- the user need not set r, but should then set eval_status 
         to a non-zero value.

  \li  4. The product \f$p = J_r(x) v\f$ of the residual Jacobian 
       \f$J_r(x)\f$ with a given
       output vector \f$v\f$ is required from the user. The vector \f$v\f$
       will be  stored in v and the product \f$Av\f$ must be returned in p,
       eval_status should be set to 0, and snls_solve_reverse_with_jacprod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but snls_solve_reverse_with_jacprod 
       should be re-entered with eval_status set to a nonzero value.

  \li  5. The product \f$J_r(x)^Tv\f$ of the transpose of the residual Jacobian
       \f$J_r(x)\f$ with a given output vector \f$v\f$ is required from 
       the user.  The vector \f$v\f$ will be  stored in v and the product
       \f$J_r(x)^Tv\f$ must be returned in p, eval_status should be set to 0, 
       and snls_solve_reverse_with_jacprod re-entered with all other arguments 
       unchanged. If the product cannot be formed, v need not be set, but 
       snls_solve_reverse_with_jacprod should be re-entered with eval_status 
       set to a nonzero value.

  \li  6. The j-th column of the residual Jacobian \f$J_r(x)\f$ is required 
       from the user, where index holds the value of j. The resulting 
       NONZEROS and their  correspinding row indices of the j-th column of
       \f$Av_o\f$ must be placed in p[0 : lp-1] and ip[0 : lp-1],
       respectively, with lp set accordingly. Additionally
       eval_status should be set to 0, and snls_solve_reverse_with_jacprod
       re-entered with all other arguments unchanged. If the column 
       cannot be formed, p, ip and lp need not be set, but 
       snls_solve_reverse_with_jacprod should be re-entered with eval_status 
       set to a nonzero value.

  \li  7. The product \f$J_r(x)v\f$ of the residual Jacobian \f$J_r(x)\f$ with 
       a  given sparse output vector \f$v\f$ is required from the user.
       The nonzero components of the vector \f$v\f$ will be stored as entries
       iv[iv_start-1:iv_end-1] of v and the product \f$Av\f$ must be returned 
       in p, eval_status should be set to 0, and snls_solve_reverse_with_jacprod
       re-entered with all other arguments unchanged; The remaining
       components of v should be ignored. If the product cannot be formed,
       v need not be set, but snls_solve_reverse_with_jacprod should be
       re-entered with eval_status set to a nonzero value.

  \li  8. A subset of the product \f$J_r(x)^Tv\f$ of the transpose of the 
       residual Jacobian \f$J_r(x)\f$ with a given output vector \f$v\f$ 
       is required from the user. The vector \f$v\f$ will be  stored in v 
       and components iv[iv_start-1:iv_end-1]
       of the product \f$J_r(x)^Tv\f$ must be returned in the relevant
       components of p (the remaining components should not be set),
       eval_status should be set to 0, and snls_solve_reverse_with_jacprod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but snls_solve_reverse_with_jacprod 
       should be re-entered with eval_status set to a nonzero value.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if the matrix products can be provided (see \c status above)

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m_r is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m_c is a scalar variable of type ipc_, that holds the number of 
   cohorts.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers.
    The i-th component of y, i = 0, ... , o-1, contains \f$y_i\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] r is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the residuals \f$r(x)\f$.
    The i-th component of r, i = 0, ... , o-1, contains \f$r_i(x)\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g(x) = J_r(x)^T r(x)\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j(x)\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] v is a one-dimensional array of size n and type rpc_, that
    is used for reverse communication (see status=2-7 above for details)

 @param[out] iv is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=5-6 above for details)

 @param[out] lvl is a scalar of type ipc_, that
    is used for reverse communication (see status=5-6 above for details)

 @param[out] lvu is a scalar of type ipc_, that
    is used for reverse communication (see status=5-6 above for details)

 @param[out] index is a scalar of type ipc_, that
    is used for reverse communication (see status=4 above for details)

 @param[in] p is a one-dimensional array of size n and type rpc_, that
    is used for reverse communication (see status=2-7 above for details)

 @param[in] ip is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=4 above for details)

 @param[in] lp is a scalar of type ipc_, that
    is used for reverse communication (see status=4 above for details)

 @param[in] w is a one-dimensional array of size m_r and type rpc_, that
    holds the values of the weights \f$w\f$. The j-th component of w, 
    j = 0, ... , n-1, contains \f$w_j\f$. It need not be set if the weights 
    are all ones, and in this case can be NULL.

*/

// *-*-*-*-*-*-*-*-*-*-    S N L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void snls_information( void **data,
                       struct snls_inform_type *inform,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see snls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S N L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void snls_terminate( void **data,
                     struct snls_control_type *control,
                     struct snls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see snls_control_type)

  @param[out] inform   is a struct containing output information
              (see snls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example snlst.c
   This is an example of how to use the package to solve a simplex-constrained
   nonlinear least-squares problem.
   A variety of supported Jacobian storage formats are shown. An example
   of preconditioning, in this case with the identity matrix which
   actually achieves nothing, is also illustrated.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example snlstf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
