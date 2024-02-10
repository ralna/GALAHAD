//* \file galahad_slls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SLLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package slls

  \section slls_intro Introduction

  \subsection slls_purpose Purpose

  This package uses a preconditioned, projected-gradient method to solve the
   <b>simplex-constrained regularized linear least-squares problem</b>
  \f[\mbox{minimize}\;\; q(x) = \frac{1}{2} \| A_o x - b\|_2^2 + \frac{1}{2} \sigma \|x\|^2\f]
\manonly
  \n
  minimize q(x) := 1/2 || A_o x - b ||^2 + sigma ||x||^2
  \n
\endmanonly
  where \f$x\f$ is required to lie in the regular simplex
  \f[e^T x = 1 \;\;\mbox{and}\;\;  x_j \geq 0, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   e^T x = 1 and x_j \[>=] 0, j = 1, ... , n,
  \n
\endmanonly
  where the \f$o\f$ by \f$n\f$ real matrix \f$A_o\f$, the vector
  \f$b\f$,and the non-negative weight
  \f$\sigma\f$ are given, and e is the vector of ones.
  Full advantage is taken of any zero
  coefficients of the design matrix \f$A_o\f$ of the <b>residuals</b>
  \f$r(x) = A x - b\f$;  the matrix need not be provided as there are options
  to obtain matrix-vector products involving \f$A_o\f$ and its transpose either
  by reverse communication or from a user-provided subroutine.

  \subsection slls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection slls_date Originally released

  October 2019, C interface July 2022.

  \subsection slls_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[e^T x = 1 \;\;\mbox{and}\;\; x \geq 0 ,\f]
\manonly
  \n
   e^T x = 1 and x [>=] 0,
  \n
\endmanonly
  the dual optimality conditions
  \f[(A_o^T A_o + \sigma I ) x = A_o^T b + z\f]
\manonly
  \n
   ( A_o^T A_o + sigma I ) x = A_o^T b + z
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

  \subsection slls_method Method

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
  \f$\alpha > 0\f$, where the projection operator \f$P(v)\f$ gives the
  nearest feasible point to \f$v\f$ within the regular simplex;
  thus this arc bends the search direction into the feasible region.
  The arc search is performed either exactly, by passing through a set
  of increasing breakpoints at which it changes direction, or inexactly,
  by evaluating a sequence of different \f$\alpha\f$  on the arc.
  All computation is designed to exploit sparsity in \f$A_o\f$.

  \subsection slls_references Reference

  Full details are provided in

  N. I. M. Gould (2022).
  Linear least-squares over the unit simplex.
  In preparation.

  \subsection slls_call_order Call order

  To solve a given problem, functions from the slls package must be called
  in the following order:

  - \link slls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link slls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - set up problem data structures and fixed values by caling one of
     - \link slls_import \endlink - in the case that \f$A_o\f$ is explicitly
        available
     - \link slls_import_without_a \endlink - in the case that only the
        effect of applying \f$A_o\f$ and its transpose to a vector is possible
  - \link slls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link slls_solve_given_a \endlink - solve the problem using values
          of \f$A_o\f$
     - \link slls_solve_reverse_a_prod \endlink - solve the problem by returning
         to the caller for products of \f$A_o\f$ and its transpose with specified
          vectors
  - \link slls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link slls_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$m\f$ by \f$n\f$ matrix \f$A_o\f$ may be presented
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

  The matrix \f$A_o\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array Ao_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_dense_column Dense column storage format

  The matrix \f$A_o\f$ is stored as a compact  dense matrix by columns, that is,
  the values of the entries of each column in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$m \ast j + i\f$  of the storage array Ao_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A_o\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays Ao_row and
  Ao_col and real array Ao_val, respectively, while the number of nonzeros
  is recorded as Ao_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A_o\f$ the i-th component of the
  integer array Ao_ptr holds the position of the first entry in this row,
  while Ao_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = Ao_ptr(i), \f$\ldots\f$, Ao_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array Ao_col, and real array Ao_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessors.

  \subsubsection unsymmetric_matrix_column_wise Sparse column-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in column j appear directly before those
  in column j+1. For the j-th column of \f$A_o\f$ the j-th component of the
  integer array Ao_ptr holds the position of the first entry in this column,
  while Ao_ptr(n) holds the total number of entries.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values \f$A_{ij}\f$
  of the  nonzero entries in the j-th column are stored in components
  l = Ao_ptr(j), \f$\ldots\f$, Ao_ptr(j+1)-1,  \f$0 \leq j \leq n-1\f$,
  of the integer array Ao_row, and real array Ao_val, respectively.
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
#ifndef GALAHAD_SLLS_H
#define GALAHAD_SLLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sbls.h"
#include "galahad_convert.h"

/**
 * control derived type as a C struct
 */
struct slls_control_type {

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
    /// assigned according to X_stat, see below), and to any other value if the
    /// values given in prob.X suffice
    ipc_ cold_start;

    /// \brief
    /// the preconditioner (scaling) used. Possible values are:
    /// /li 0. no preconditioner.
    /// /li 1. a diagonal preconditioner that normalizes the rows of \f$A_o\f$.
    /// /li anything else. a preconditioner supplied by the user either via
    ///     a subroutine call of eval_prec} or via reverse communication.
    ipc_ preconditioner;

    /// \brief
    /// the ratio of how many iterations use CGLS rather than steepest descent
    ipc_ ratio_cg_vs_sd;

    /// \brief
    /// the maximum number of per-iteration changes in the working set
    /// permitted when allowing CGLS rather than steepest descent
    ipc_ change_max;

    /// \brief
    /// how many CG iterations to perform per SLLS iteration
    /// (-ve reverts to n+1)
    ipc_ cg_maxit;

    /// \brief
    /// the maximum number of steps allowed in a piecewise arcsearch (-ve=infini
    ipc_ arcsearch_max_steps;

    /// \brief
    /// the unit number to write generated SIF file describing the current probl
    ipc_ sif_file_device;

    /// \brief
    /// the objective function will be regularized by adding 1/2 weight ||x||^2
    rpc_ weight;

    /// \brief
    /// the required accuracy for the dual infeasibility
    rpc_ stop_d;

    /// \brief
    /// the CG iteration will be stopped as soon as the current norm of the
    /// preconditioned gradient is smaller than
    /// max( stop_cg_relative * initial preconditioned gradient,
    /// stop_cg_absolute)
    rpc_ stop_cg_relative;
    rpc_ stop_cg_absolute;

    /// \brief
    /// the largest permitted arc length during the piecewise line search
    rpc_ alpha_max;

    /// \brief
    /// the initial arc length during the inexact piecewise line search
    rpc_ alpha_initial;

    /// \brief
    /// the arc length reduction factor for the inexact piecewise line search
    rpc_ alpha_reduction;

    /// \brief
    /// the required relative reduction during the inexact piecewise line search
    rpc_ arcsearch_acceptance_tol;

    /// \brief
    /// the stabilisation weight added to the search-direction subproblem
    rpc_ stabilisation_weight;

    /// \brief
    /// the maximum CPU time allowed (-ve = no limit)
    rpc_ cpu_time_limit;

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
struct slls_time_type {

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
struct slls_inform_type {

    /// \brief
    /// reported return status.
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
    struct slls_time_type time;

    /// \brief
    /// inform values from SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// inform values for CONVERT
    struct convert_inform_type convert_inform;
};

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void slls_initialize( void **data,
                     struct slls_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see slls_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    B L L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void slls_read_specfile( struct slls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNSLLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/slls.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see slls_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    B L L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void slls_import( struct slls_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char Ao_type[],
                 ipc_ Ao_ne,
                 const ipc_ Ao_row[],
                 const ipc_ Ao_col[],
                 ipc_ Ao_ptr_ne,
                 const ipc_ Ao_ptr[] );


/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see slls_control_type)

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
  \li -3. The restrictions n > 0, o > 0 or requirement that type contains
       its relevant string 'coordinate', 'sparse_by_rows',
       'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns';
       has been violated.

@param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

@param[in] o is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in]  Ao_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the objective design matrix, \f$A_o\f$. It should be one of
   'coordinate', 'sparse_by_rows', 'sparse_by_columns',
   'dense' or 'dense_by_columns'; lower or upper case variants are allowed.

 @param[in]  Ao_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A_o\f$ in the sparse storage schemes.
   It need not be set for either of the dense schemes.

 @param[in]  Ao_row is a one-dimensional array of size Ao_ne and type ipc_, that
   holds the row indices of \f$A_o\f$ in the sparse co-ordinate
   or sparse column-wise storage schemes. It need not be set for any of the
   other schemes, and in this case can be NULL.

 @param[in]  Ao_col is a one-dimensional array of size Ao_ne and type ipc_,
   that holds the column indices of \f$A_o\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  Ao_ptr_ne is a scalar variable of type ipc_, that holds the
   length of the pointer array if sparse row or column storage scheme is
   used for \f$A_o\f$. For the sparse row scheme,  Ao_ptr_ne should be at least
   o+1, while for the sparse column scheme,  it should be at least n+1,
   It need not be set when the other schemes are used.

 @param[in]  Ao_ptr is a one-dimensional array of size o+1 and type ipc_,
   that holds the starting position of each row of \f$A_o\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   By contrast, it is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each column of \f$A_o\f$, as well as the
   total number of entries, in the sparse column-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.

*/

// *-*-*-*-*-*-*-    B L L S  _ I M P O R T _ W I T H O U T _ A   -*-*-*-*-*-*

void slls_import_without_a( struct slls_control_type *control,
                            void **data,
                            ipc_ *status,
                            ipc_ n,
                            ipc_ o );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see slls_control_type)

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
  \li -3. The restriction n > 0 or o > 0 has been violated.

@param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

@param[in] o is a scalar variable of type ipc_, that holds the number of
    residuals.

*/
// *-*-*-*-*-*-*-    B L L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void slls_reset_control( struct slls_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see slls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-   B L L S _ S O L V E _ G I V E N _ A   -*-*-*-*-*-*-*-*-

void slls_solve_given_a( void **data,
                         void *userdata,
                         ipc_ *status,
                         ipc_ n,
                         ipc_ o,
                         ipc_ Ao_ne,
                         const rpc_ Ao_val[],
                         const rpc_ b[],
                         rpc_ x[],
                         rpc_ z[],
                         rpc_ r[],
                         rpc_ g[],
                         ipc_ x_stat[],
                         ipc_ (*eval_prec)(
                              ipc_, const rpc_[],
                              rpc_[], const void * ) );

/*!<
 Solve the bound-constrained linear least-squares problem when the
 Jacobian \f$A_o\f$ is available.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

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
  \li -3. The restrictions n > 0, o > 0 or requirement that a type contains
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

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

@param[in] o is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] Ao_ne is a scalar variable of type ipc_, that holds the number of
    entries in the design matrix \f$A_o\f$.

 @param[in] Ao_val is a one-dimensional array of size A_ne and type rpc_,
    that holds the values of the entries of the designmatrix \f$A_o\f$
    in any of the available storage schemes.

 @param[in] b is a one-dimensional array of size o and type rpc_, that
    holds the constant term \f$b\f$ in the residuals.
    The i-th component of b, i = 0, ... ,  m-1, contains  \f$b_i \f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] r is a one-dimensional array of size o and type rpc_, that
    holds the values of the residuals \f$r = A x - b\f$.
    The i-th component of r, i = 0, ... , m-1, contains \f$r_i\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g = A_o^T r\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param  eval_prec is an optional user-supplied function that may be NULL.
   If non-NULL, it must have the following signature:
   \code
       ipc_ eval_prec( ipc_ n, const rpc_ v[], rpc_ p[],
                      const void *userdata )
   \endcode
   The product \f$p = P^{-1} v\f$ involving the user's preconditioner \f$P\f$
   with the vector v = \f$v\f$, the result \f$p\f$
   must be retured in p, and the function return value set to 0. If the
   evaluation is impossible, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata.

*/

//  *-*-*-*-*-*-   B L L S _ S O L V E _ R E V E R S E _ A _ P R O D   -*-*-*-*-

void slls_solve_reverse_a_prod( void **data,
                                ipc_ *status,
                                ipc_ *eval_status,
                                ipc_ n,
                                ipc_ o,
                                const rpc_ b[],
                                rpc_ x[],
                                rpc_ z[],
                                rpc_ r[],
                                rpc_ g[],
                                ipc_ x_stat[],
                                rpc_ v[],
                                const rpc_ p[],
                                ipc_ nz_v[],
                                ipc_ *nz_v_start,
                                ipc_ *nz_v_end,
                                const ipc_ nz_p[],
                                ipc_ nz_p_end );

/*!<
 Solve the bound-constrained linear least-squares problem when the
 products of the Jacobian \f$A_o\f$ and its transpose
 with specified vectors may be computed by the calling program.

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
  \li  2. The product \f$Av\f$ of the residual Jacobian \f$A_o\f$ with a given
       output vector \f$v\f$ is required from the user. The vector \f$v\f$
       will be  stored in v and the product \f$Av\f$ must be returned in p,
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but slls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  3. The product \f$A_o^Tv\f$ of the transpose of the residual Jacobian
       \f$A_o\f$ with a given output vector \f$v\f$ is required from the user.
       The vector \f$v\f$ will be  stored in v and the product
       \f$A_o^Tv\f$ must be returned in p,
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but slls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  4. The product \f$Av\f$ of the residual Jacobian \f$A_o\f$ with a given
       sparse output vector \f$v\f$ is required from the user.
       The nonzero components of the vector \f$v\f$ will be stored as entries
          nz_in[nz_in_start-1:nz_in_end-1]
       of v and the product \f$Av\f$ must be returned in p,
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged; The remaining
       components of v should be ignored. If the product cannot
       be formed, v need not be set, but slls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  5. The nonzero components of the product \f$Av\f$ of the residual
       Jacobian \f$A_o\f$ with a given sparse output vector \f$v\f$ is required
       from the user. The nonzero components of the vector \f$v\f$ will be
       stored as entries
          nz_in[nz_in_start-1:nz_in_end-1]
       of v; the remaining components of v should be ignored.
       The resulting <b>nonzeros</b> in the product \f$Av\f$
       must be placed in their appropriate comnponents of p, while a list
       of indices of the nonzeros placed in
         nz_out[0 : nz_out_end-1]
       and the number of nonzeros recorded in nz_out_end. Additionally,
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v, nz_out_end and nz_out  need not be set, but
       slls_solve_reverse_a_prod should be re-entered with eval_status set
       to a nonzero value.

  \li  6. A subset of the product \f$A_o^Tv\f$ of the transpose of the residual
       Jacobian
       \f$A_o\f$ with a given output vector \f$v\f$ is required from the user.
       The vector \f$v\f$ will be  stored in v and components
          nz_in[nz_in_start-1:nz_in_end-1]
       of the product \f$A_o^Tv\f$ must be returned in the relevant
       components of p (the remaining components should not be set),
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but slls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.

  \li  7. The product \f$P^{-1}v\f$ of the inverse of the preconditioner
       \f$P\f$ with a given output vector \f$v\f$ is required from the user.
       The vector \f$v\f$ will be  stored in v and the product \f$P^{-1} v\f$
       must be returned in p,
       status_eval should be set to 0, and slls_solve_reverse_a_prod
       re-entered with all other arguments unchanged. If the product cannot
       be formed, v need not be set, but slls_solve_reverse_a_prod should be
       re-entered with eval_status set to a nonzero value.
       This value of status can only occur if the user has set
       control.preconditioner = 2.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if the matrix products can be provided (see \c status above)

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] o is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] b is a one-dimensional array of size o and type rpc_, that
    holds the constant term \f$b\f$ in the residuals.
    The i-th component of b, i = 0, ... ,  m-1, contains  \f$b_i \f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] r is a one-dimensional array of size o and type rpc_, that
    holds the values of the residuals \f$r = A x - b\f$.
    The i-th component of c, i = 0, ... , o-1, contains \f$c_i\f$.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the values of the gradient \f$g = A_o^T r\f$.
    The j-th component of g, j = 0, ... , n-1, contains \f$g_j\f$.

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

 @param[in] p is a one-dimensional array of size n and type rpc_, that
    is used for reverse communication (see status=2-4 above for details)

 @param[out] nz_v is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_start is a scalar of type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[out] nz_v_end is a scalar of type ipc_, that
    is used for reverse communication (see status=3-4 above for details)

 @param[in] nz_p is a one-dimensional array of size n and type ipc_, that
    is used for reverse communication (see status=4 above for details)

 @param[in] nz_p_end is a scalar of type ipc_, that
    is used for reverse communication (see status=4 above for details)

*/

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void slls_information( void **data,
                      struct slls_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see slls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    B L L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void slls_terminate( void **data,
                    struct slls_control_type *control,
                    struct slls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see slls_control_type)

  @param[out] inform   is a struct containing output information
              (see slls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example sllst.c
   This is an example of how to use the package to solve a bound-constrained
   linear least-squares problem.
   A variety of supported Jacobian storage formats are shown. An example
   of preconditioning, in this case with the identity matrix which
   actually achieves nothing, is also illustrated.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example sllstf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
