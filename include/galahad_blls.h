//* \file galahad_blls.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-02-21 AT 12:41 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BLLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package blls
 
  \section blls_intro Introduction

  \subsection blls_purpose Purpose

  This package uses a preconditioned, projected-gradient method to solve the 
   <b>bound-constrained regularized linear least-squares problem</b>
  \f[\mbox{minimize}\;\; q(x) = q(x) = \frac{1}{2} \| A x - b\|_2^2 + \frac{1}{2} \sigma \|x\|^2\f]
\manonly
  \n
  minimize q(x) := 1/2 || A x - b ||^2 + sigma ||x||^2
  \n
\endmanonly
  subject to the simple bound constraints
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the \f$m\f$ by \f$n\f$ real matrix \f$A\f$, the vectors 
  \f$b\f$, \f$x^{l}\f$, \f$x^{u}\f$ and the non-negative weight 
  \f$\sigma\f$ are given.   Any of the constraint bounds \f$x_j^l\f$ and 
  \f$x_j^u\f$ may be infinite.  Full advantage is taken of any zero 
  coefficients of the Jacobian matrix \f$A\f$ of the <b>residuals</b> 
  \f$c(x) = A x - b\f$;  the matrix need not be provided as there are options 
  to obtain matrix-vector products involving \f$A\f$ and its transpose either 
  by reverse communication or from a user-provided subroutine.

  \subsection blls_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection blls_date Originally released

  October 2019, C interface March 2022.

  \subsection blls_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[x^l \leq x \leq x^u,\f]
\manonly
  \n
   x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[(A^T A + \sigma I ) x = A^T b + z\f]
\manonly
  \n
   ( A^T A + sigma I ) x = A^T b + z 
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

  \subsection blls_method Method

  The method is iterative. Each iteration proceeds in two stages.
  Firstly, a search direction \f$s\f$ from the current estimate of the solution
  \f$x\f$ is computed. This may be in a scaled steepest-descent direction, or,
  if the working set of variables on bounds has not changed dramatically,
  in a direction that provides an approximate minimizer of the objective
  over a subspace comprising the currently free-variables. The latter is
  computed either using an appropriate sparse factorization by the
  GALAHAD package SBLS, or by the  conjugate-gradient least-squares (CGLS) 
  method; tt may be necessary to regularize the subproblem very slightly to 
  avoid a ill-posedness. Thereafter, a piecewise linesearch (arc search) is 
  carried out along the arc \f$x(\alpha) = P( x + \alpha s)\f$ for 
  \f$\alpha > 0\f$, where the projection operator is defined component-wise 
  at any feasible point \f$v\f$ to be
  \f[P_j(v) = \min( \max( x_j, x_j^l), x_j^u);\f]
  thus this arc bends the search direction into the feasible region.
  The arc search is performed either exactly, by passing through a set
  of increasing breakpoints at which it changes direction, or inexactly,
  by evaluating a sequence of different \f$\alpha\f$  on the arc.
  All computation is designed to exploit sparsity in \f$A\f$.

  \subsection blls_references Reference

  Full details are provided in

  N. I. M. Gould (2022).
  Numerical methods for solving bound-constrained linear least squares problems.
  In preparation.

  \subsection blls_call_order Call order
  To solve a given problem, functions from the blls package must be called 
  in the following order:

  - \link blls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link blls_read_specfile \endlink (optional) - override control values 
      by reading replacement values from a file
  - set up problem data structures and fixed values by caling one of
     - \link blls_import \endlink - in the case that \f$A\f$ is explicitly 
        available
     - \link blls_import_without_a \endlink - in the case that only the
        effect of applying \f$A\f$ and its transpose to a vector is possible
  - \link blls_reset_control \endlink (optional) - possibly change control 
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of 
     - \link blls_solve_given_a \endlink - solve the problem using values
          of \f$A\f$
     - \link blls_solve_reverse_a_prod \endlink - solve the problem by returning
         to the caller for products of \f$A\f$ and its transpose with specified
          vectors
  - \link blls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link blls_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$m\f$ by \f$n\f$ matrix \f$A\f$ may be presented 
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
  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is, 
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$, 
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_dense_column Dense column storage format
  The matrix \f$A\f$ is stored as a compact  dense matrix by columns, that is, 
  the values of the entries of each column in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$m \ast j + i\f$  of the storage array A_val
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
  while A_ptr(m) holds the total number of entries plus one.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values 
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array A_col, and real array A_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessors.

  \subsubsection unsymmetric_matrix_column_wise Sparse column-wise storage format
  Again only the nonzero entries are stored, but this time
  they are ordered so that those in column j appear directly before those
  in column j+1. For the j-th column of \f$A\f$ the j-th component of the
  integer array A_ptr holds the position of the first entry in this column,
  while A_ptr(n) holds the total number of entries plus one.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values \f$A_{ij}\f$ 
  of the  nonzero entries in the j-th column are stored in components
  l = A_ptr(j), \f$\ldots\f$, A_ptr(j+1)-1,  \f$0 \leq j \leq n-1\f$,
  of the integer array A_row, and real array A_val, respectively.
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
#ifndef GALAHAD_BLLS_H 
#define GALAHAD_BLLS_H

// precision
#include "galahad_precision.h"

// required packages
#include "galahad_sbls.h"
#include "galahad_convert.h"

/**
 * control derived type as a C struct
 */
struct blls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit number for error and warning diagnostics
    int error;

    /// \brief
    /// general output unit number
    int out;

    /// \brief
    /// the level of output required
    int print_level;

    /// \brief
    /// on which iteration to start printing
    int start_print;

    /// \brief
    /// on which iteration to stop printing
    int stop_print;

    /// \brief
    /// how many iterations between printing
    int print_gap;

    /// \brief
    /// how many iterations to perform (-ve reverts to HUGE(1)-1)
    int maxit;

    /// \brief
    /// cold_start should be set to 0 if a warm start is required (with variable
    /// assigned according to X_stat, see below), and to any other value if the
    /// values given in prob.X suffice
    int cold_start;

    /// \brief
    /// the preconditioner (scaling) used. Possible values are:
    /// /li 0. no preconditioner.
    /// /li 1. a diagonal preconditioner that normalizes the rows of \f$A\f$.
    /// /li anything else. a preconditioner supplied by the user either via 
    ///     a subroutine call of eval_prec} or via reverse communication.
    int preconditioner;

    /// \brief
    /// the ratio of how many iterations use CGLS rather than steepest descent
    int ratio_cg_vs_sd;

    /// \brief
    /// the maximum number of per-iteration changes in the working set
    /// permitted when allowing CGLS rather than steepest descent
    int change_max;

    /// \brief
    /// how many CG iterations to perform per BLLS iteration 
    /// (-ve reverts to n+1)
    int cg_maxit;

    /// \brief
    /// the maximum number of steps allowed in a piecewise arcsearch (-ve=infini
    int arcsearch_max_steps;

    /// \brief
    /// the unit number to write generated SIF file describing the current probl
    int sif_file_device;

    /// \brief
    /// the objective function will be regularized by adding 1/2 weight ||x||^2
    real_wp_ weight;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    real_wp_ infinity;

    /// \brief
    /// the required accuracy for the dual infeasibility
    real_wp_ stop_d;

    /// \brief
    /// any pair of constraint bounds (x_l,x_u) that are closer than
    /// identical_bounds_tol will be reset to the average of their values
    ///
    real_wp_ identical_bounds_tol;

    /// \brief
    /// the CG iteration will be stopped as soon as the current norm of the
    /// preconditioned gradient is smaller than
    /// max( stop_cg_relative * initial preconditioned gradient, 
    /// stop_cg_absolute)
    real_wp_ stop_cg_relative;
    real_wp_ stop_cg_absolute;

    /// \brief
    /// the largest permitted arc length during the piecewise line search
    real_wp_ alpha_max;

    /// \brief
    /// the initial arc length during the inexact piecewise line search
    real_wp_ alpha_initial;

    /// \brief
    /// the arc length reduction factor for the inexact piecewise line search
    real_wp_ alpha_reduction;

    /// \brief
    /// the required relative reduction during the inexact piecewise line search
    real_wp_ arcsearch_acceptance_tol;

    /// \brief
    /// the stabilisation weight added to the search-direction subproblem
    real_wp_ stabilisation_weight;

    /// \brief
    /// the maximum CPU time allowed (-ve = no limit)
    real_wp_ cpu_time_limit;

    /// \brief
    /// direct_subproblem_solve is true if the least-squares subproblem is to be
    /// solved using a matrix factorization, and false if conjugate gradients
    /// are to be preferred
    bool direct_subproblem_solve;

    /// \brief
    /// exact_arc_search is true if an exact arc_search is required, and false
    /// if an approximation suffices
    bool exact_arc_search;

    /// \brief
    /// advance is true if an inexact exact arc_search can increase steps as 
    /// well as decrease them
    bool advance;

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

    /// \brief
    /// control parameters for CONVERT
    struct convert_control_type convert_control;
};

/**
 * time derived type as a C struct
 */
struct blls_time_type {

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
struct blls_inform_type {

    /// \brief
    /// reported return status. 
    int status;

    /// \brief
    /// Fortran STAT value after allocate failure
    int alloc_status;

    /// \brief
    /// status return from factorization
    int factorization_status;

    /// \brief
    /// number of iterations required
    int iter;

    /// \brief
    /// number of CG iterations required
    int cg_iter;

    /// \brief
    /// current value of the objective function
    real_wp_ obj;

    /// \brief
    /// current value of the projected gradient
    real_wp_ norm_pg;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// times for various stages
    struct blls_time_type time;

    /// \brief
    /// inform values from SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// inform values for CONVERT
    struct convert_inform_type convert_inform;
};

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void blls_initialize( void **data, 
                     struct blls_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see blls_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    B L L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void blls_read_specfile( struct blls_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNBLLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/blls.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information 
              (see blls_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    B L L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void blls_import( struct blls_control_type *control,
                 void **data,
                 int *status,
                 int n,
                 int m,
                 const char A_type[], 
                 int A_ne, 
                 const int A_row[],
                 const int A_col[], 
                 const int A_ptr[] );


/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see blls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restrictions n > 0, m > 0 or requirement that type contains
       its relevant string 'coordinate', 'sparse_by_rows',
       'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns';
       has been violated.

@param[in] n is a scalar variable of type int, that holds the number of
    variables.

@param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in] A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices symmetric storage scheme \endlink 
   used for the Jacobian \f$A\f$. It should be one of 
   'coordinate', 'sparse_by_rows', 'sparse_by_columns', 
   'dense_by_rows', or 'dense_by_columns';
   lower or upper case variants are allowed.

 @param[in] A_ne is a scalar variable of type int, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate storage scheme. 
   It need not be set for any of the other schemes.

 @param[in] A_row is a one-dimensional array of size A_ne and type int, that 
   holds the row indices of \f$A\f$ in the sparse co-ordinate 
   or sparse column-wise storage scheme. It need not be set for any of 
   the other schemes, and in this case can be NULL.

 @param[in] A_col is a one-dimensional array of size A_ne and type int,
   that holds the column indices of \f$A\f$ in either the sparse co-ordinate, 
   or the sparse row-wise storage scheme. It need not be set for any of 
   the other schemes, and in this case can be NULL.

 @param[in] A_ptr is a one-dimensional array of size n+1 or m+1 and type int,
   that holds the starting position of each row of \f$A\f$, as well as the 
   total number of entries plus one, in the sparse row-wise storage scheme, or
   the starting position of each column of \f$A\f$, as well as the 
   total number of entries plus one, in the sparse column-wise storage scheme.
   It need not be set when the other schemes are used, 
   and in this case can be NULL.

*/

// *-*-*-*-*-*-*-    B L L S  _ I M P O R T _ W I T H O U T _ A   -*-*-*-*-*-*

void blls_import_without_a( struct blls_control_type *control,
                            void **data,
                            int *status,
                            int n,
                            int m );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see blls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or m > 0 has been violated.

@param[in] n is a scalar variable of type int, that holds the number of
    variables.

@param[in] m is a scalar variable of type int, that holds the number of
    residuals.

*/
// *-*-*-*-*-*-*-    B L L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void blls_reset_control( struct blls_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see blls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-   B L L S _ S O L V E _ G I V E N _ A   -*-*-*-*-*-*-*-*-

void blls_solve_given_a( void **data,
                         void *userdata, 
                         int *status,
                         int n, 
                         int m,
                         int A_ne,
                         const real_wp_ A_val[], 
                         const real_wp_ b[], 
                         const real_wp_ x_l[], 
                         const real_wp_ x_u[], 
                         real_wp_ x[], 
                         real_wp_ z[], 
                         real_wp_ c[], 
                         real_wp_ g[], 
                         int x_stat[],
                         int (*eval_prec)(
                              int, const real_wp_[], 
                              real_wp_[], const void * ) );

/*!<
 Solve the bound-constrained linear least-squares problem when the 
 Jacobian \f$A\f$ is available.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restrictions n > 0, m > 0 or requirement that a type contains
       its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns',
       'dense_by_rows' or 'dense_by_columns' has been violated.
  \li -4. The simple-bound constraints are inconsistent.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
 
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

@param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in] A_ne is a scalar variable of type int, that holds the number of 
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[in] A_val is a one-dimensional array of size A_ne and type double, 
    that holds the values of the entries of the lower triangular part of the 
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] b is a one-dimensional array of size m and type double, that 
    holds the constant term \f$b\f$ in the residuals.
    The i-th component of b, i = 0, ... ,  m-1, contains  \f$b_i \f$.
  
 @param[in] x_l is a one-dimensional array of size n and type double, that 
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.
  
 @param[in] x_u is a one-dimensional array of size n and type double, that 
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.
  
 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in,out] z is a one-dimensional array of size n and type double, that 
    holds the values \f$z\f$ of the dual variables. 
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.
  
 @param[out] c is a one-dimensional array of size m and type double, that 
    holds the values of the residuals \f$c = A x - b\f$.
    The i-th component of c, i = 0, ... , m-1, contains \f$c_i\f$.
  
 @param[out] g is a one-dimensional array of size n and type double, that 
    holds the values of the gradient \f$g = A^T c\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j\f$.
  
 @param[in,out] x_stat is a one-dimensional array of size n and type int, that 
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param  eval_prec is an optional user-supplied function that may be NULL. 
   If non-NULL, it must have the following signature:
   \code
       int eval_prec( int n, const double v[], double p[], 
                      const void *userdata )
   \endcode
   The product \f$p = P^{-1} v\f$ involving the user's preconditioner \f$P\f$ 
   with the vector v = \f$v\f$, the result \f$p\f$ 
   must be retured in p, and the function return value set to 0. If the 
   evaluation is impossible, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata. 

*/  

//  *-*-*-*-*-*-   B L L S _ S O L V E _ R E V E R S E _ A _ P R O D   -*-*-*-*-

void blls_solve_reverse_a_prod( void **data,
                                int *status,
                                int *eval_status, 
                                int n, 
                                int m,
                                const real_wp_ b[], 
                                const real_wp_ x_l[], 
                                const real_wp_ x_u[], 
                                real_wp_ x[], 
                                real_wp_ z[], 
                                real_wp_ c[], 
                                real_wp_ g[], 
                                int x_stat[],
                                real_wp_ v[], 
                                const real_wp_ p[],
                                int nz_v[], 
                                int *nz_v_start, 
                                int *nz_v_end,
                                const int nz_p[], 
                                int nz_p_end );

/*!<
 Solve the bound-constrained linear least-squares problem when the 
 products of the Jacobian \f$A\f$ and its transpose
 with specified vectors may be computed by the calling program.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li  2. The product \f$Av\f$ of the residual Jacobian \f$A\f$ with a given 
       output vector \f$v\f$ is required from the user. The vector \f$v\f$ 
       will be  stored in v and the product \f$Av\f$ must be returned in p,
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but blls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  3. The product \f$A^Tv\f$ of the transpose of the residual Jacobian 
       \f$A\f$ with a given output vector \f$v\f$ is required from the user. 
       The vector \f$v\f$ will be  stored in v and the product 
       \f$A^Tv\f$ must be returned in p,
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but blls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  4. The product \f$Av\f$ of the residual Jacobian \f$A\f$ with a given 
       sparse output vector \f$v\f$ is required from the user. 
       The nonzero components of the vector \f$v\f$ will be stored as entries 
          nz_in[nz_in_start-1:nz_in_end-1]
       of v and the product \f$Av\f$ must be returned in p,
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged; The remaining 
       components of v should be ignored. If the product cannot
       be formed, v need not be set, but blls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  5. The nonzero components of the product \f$Av\f$ of the residual 
       Jacobian \f$A\f$ with a given sparse output vector \f$v\f$ is required 
       from the user. The nonzero components of the vector \f$v\f$ will be 
       stored as entries 
          nz_in[nz_in_start-1:nz_in_end-1]
       of v; the remaining components of v should be ignored. 
       The resulting <b>nonzeros</b> in the product \f$Av\f$ 
       must be placed in their appropriate comnponents of p, while a list 
       of indices of the nonzeros placed in 
         nz_out[0 : nz_out_end-1]
       and the number of nonzeros recorded in nz_out_end. Additionally,
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v, nz_out_end and nz_out  need not be set, but 
       blls_solve_reverse_a_prod should be re-entered with eval_status set 
       to a nonzero value.

  \li  6. A subset of the product \f$A^Tv\f$ of the transpose of the residual 
       Jacobian 
       \f$A\f$ with a given output vector \f$v\f$ is required from the user. 
       The vector \f$v\f$ will be  stored in v and components
          nz_in[nz_in_start-1:nz_in_end-1]
       of the product \f$A^Tv\f$ must be returned in the relevant 
       components of p (the remaining components should not be set),
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but blls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  7. The product \f$P^{-1}v\f$ of the inverse of the preconditioner
       \f$P\f$ with a given output vector \f$v\f$ is required from the user. 
       The vector \f$v\f$ will be  stored in v and the product \f$P^{-1} v\f$ 
       must be returned in p,
       status_eval should be set to 0, and blls_solve_reverse_a_prod 
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but blls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.
       This value of status can only occur if the user has set 
       control.preconditioner = 2.

 @param[in,out] eval_status is a scalar variable of type int, that is used to 
    indicate if the matrix products can be provided (see \c status above) 
  
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in] b is a one-dimensional array of size m and type double, that 
    holds the constant term \f$b\f$ in the residuals.
    The i-th component of b, i = 0, ... ,  m-1, contains  \f$b_i \f$.
  
 @param[in] x_l is a one-dimensional array of size n and type double, that 
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.
  
 @param[in] x_u is a one-dimensional array of size n and type double, that 
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.
  
 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[out] c is a one-dimensional array of size m and type double, that 
    holds the values of the residuals \f$c = A x - b\f$.
    The i-th component of c, i = 0, ... , m-1, contains \f$c_i\f$.
  
 @param[out] g is a one-dimensional array of size n and type double, that 
    holds the values of the gradient \f$g = A^T c\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j\f$.
  
 @param[in,out] z is a one-dimensional array of size n and type double, that 
    holds the values \f$z\f$ of the dual variables. 
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.
  
 @param[in,out] x_stat is a one-dimensional array of size n and type int, that 
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] v is a one-dimensional array of size n and type double, that 
    is used for reverse communication (see status=2-4 above for details)

 @param[in] p is a one-dimensional array of size n and type double, that 
    is used for reverse communication (see status=2-4 above for details)

 @param[out] nz_v is a one-dimensional array of size n and type int, that 
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_start is a scalar of type int, that 
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_end is a scalar of type int, that 
    is used for reverse communication (see status=3-4 above for details)

 @param[in] nz_p is a one-dimensional array of size n and type int, that 
    is used for reverse communication (see status=4 above for details)

 @param[in] nz_p_end is a scalar of type int, that 
    is used for reverse communication (see status=4 above for details)

*/  

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void blls_information( void **data,
                      struct blls_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see blls_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void blls_terminate( void **data, 
                    struct blls_control_type *control, 
                    struct blls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see blls_control_type)

  @param[out] inform   is a struct containing output information
              (see blls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example bllst.c
   This is an example of how to use the package to solve a bound-constrained
   linear least-squares problem.
   A variety of supported Jacobian storage formats are shown. An example
   of preconditioning, in this case with the identity matrix which
   actually achieves nothing, is also illustrated.
  
   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example bllstf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
