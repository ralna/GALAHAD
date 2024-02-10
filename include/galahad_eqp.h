//* \file galahad_eqp.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_EQP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 7th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package eqp

  \section eqp_intro Introduction

  \subsection eqp_purpose Purpose

  This package uses an iterative method to solve the
  <b>equality-constrained quadratic programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  subject to the linear constraints
  \f[(1) \;\; A x + c = 0,\f]
  where the \f$n\f$ by \f$n\f$ symmetric matrix \f$H\f$,
  the \f$m\f$ by \f$n\f$ matrix \f$A\f$, the vectors \f$g\f$ and \f$c\f$
  Full advantage is taken of any zero coefficients in the matrices \f$H\f$
  and \f$A\f$.

  The package may alternatively be used to minimize the (shifted) squared-
  least-distance objective
  \f[\frac{1}{2} \sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2 + g^T x + f,\f]
\manonly
  \n
   minimize 1/2 \sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2  + g^T x + f,
  \n
\endmanonly
  subject to the linear constraint (1), for given vectors \f$w\f$ and
  \f$x^0\f$.

  \subsection eqp_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection eqp_date Originally released

  March 2006, C interface January 2021.

  \subsection eqp_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[(2) \;\; A x + c = 0\f]
\manonly
  \n
  (2) A x + c = 0
  \n
\endmanonly
  and the dual optimality conditions
\latexonly
  \[  H x + g - A^T y = 0 \;\; (\mbox{or} \;\; W^{2} (x -x^0) + g - A^T y = 0 \;\;  \mbox{for the shifted-least-distance type objective})\]
\endlatexonly
\htmlonly
  $$H x + g - A^T y = 0 \;\; (\mbox{or} \;\;  W^{2} (x -x^0) + g - A^T y = 0 \;\; \mbox{for the shifted-least-distance type objective})$$
\endhtmlonly
\manonly
  \n
  (3) H x + g - A^T y = 0
       (or W^2 (x -x^0) + g - A^T y = 0
        for the shifted-least-distance type objective)
  \n
\endmanonly
  where the diagonal matrix \f$W^2\f$ has diagonal entries \f$w_j^2\f$,
  \f$j = 1, \ldots , n\f$, and where the vector \f$y\f$ is
  known as the Lagrange multipliers for the linear constraints.

 \subsection eqp_method Method

  A solution to the problem is found in two phases.
  In the first, a point \f$x_F\f$ satisfying (2) is found.
  In the second, the required solution \f$x = x_F + s\f$
  is determined by finding \f$s\f$ to minimize
  \f$q(s) = \frac{1}{2} s^T H s + g_F^T s + f_F\f$
  subject to the homogeneous constraints \f$A s = zero\f$,
  where \f$g_F = H x_F + g\f$ and
  \f$f_F = \frac{1}{2} x_F^T H x_F + g^T x_F + f\f$.
  The required constrained minimizer of \f$q(s)\f$ is obtained
  by implictly applying the preconditioned conjugate-gradient method
  in the null space of \f$A\f$. Any preconditioner of the form
  \f[ K_G = \mat{cc}{ G & A^T \\ A  & 0 }\f]
\manonly
  \n
  K_G = ( G  A^T )
        ( A   0  )
  \n
\endmanonly
  is suitable, and the GALAHAD package SBLS provides a number of
  possibilities. In order to ensure that the minimizer obtained is
  finite, an additional, precautionary trust-region constraint \f$\|s\|
  \leq \Delta\f$ for some suitable positive radius \f$\Delta\f$ is
  imposed, and the GALAHAD package GLTR is used to solve this
  additionally-constrained problem.

  \subsection eqp_references Reference

  The preconditioning aspcets are described in detail in

  H. S. Dollar, N. I. M. Gould and A. J. Wathen.
  ``On implicit-factorization constraint preconditioners''.
  In  Large Scale Nonlinear Optimization (G. Di Pillo and M. Roma, eds.)
  Springer Series on Nonconvex Optimization and Its Applications, Vol. 83,
  Springer Verlag (2006) 61-82

  and

  H. S. Dollar, N. I. M. Gould, W. H. A. Schilders and A. J. Wathen
  ``On iterative methods and implicit-factorization preconditioners for
  regularized saddle-point systems''.
  SIAM Journal on Matrix Analysis and Applications, <b>28(1)</b> (2006)
  170-189,

  while the constrained conjugate-gradient method is discussed in

  N. I. M. Gould, S. Lucidi, M. Roma and Ph. L. Toint,
  Solving the trust-region subproblem using the Lanczos method.
  SIAM Journal on Optimization <b>9:2</b> (1999), 504-525.

  \subsection eqp_call_order Call order

  To solve a given problem, functions from the eqp package must be called
  in the following order:

  - \link eqp_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link eqp_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link eqp_import \endlink - set up problem data structures and fixed
      values
  - \link eqp_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link eqp_solve_qp \endlink - solve the quadratic program
     - \link eqp_solve_sldqp \endlink - solve the shifted least-distance problem
  - \link eqp_resolve_qp \endlink (optional) - resolve the problem with the
    same Hessian and Jacobian, but different \f$g\f$, \f$f\f$ and/or \f$c\f$
  - \link eqp_information \endlink (optional) - recover information about
    the solution and solution process
  - \link eqp_terminate \endlink - deallocate data structures

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
  \f$H\f$ may be presented
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

  \subsubsection symmetric_matrix_scaled_identity Multiples of the identity storage format

  If \f$H\f$ is a multiple of the identity matrix, (i.e., \f$H = \alpha I\f$
  where \f$I\f$ is the n by n identity matrix and \f$\alpha\f$ is a scalar),
  it suffices to store \f$\alpha\f$ as the first component of H_val.

  \subsubsection symmetric_matrix_identity The identity matrix format

  If \f$H\f$ is the identity matrix, no values need be stored.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_EQP_H
#define GALAHAD_EQP_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_fdc.h"
#include "galahad_sbls.h"
#include "galahad_gltr.h"

/**
 * control derived type as a C struct
 */
struct eqp_control_type {

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
    /// the level of output required is specified by print_level
    ipc_ print_level;

    /// \brief
    /// the factorization to be used. Possible values are
    /// /li 0  automatic
    /// /li 1  Schur-complement factorization
    /// /li 2  augmented-system factorization                         (OBSOLETE)
    ipc_ factorization;

    /// \brief
    /// the maximum number of nonzeros in a column of A which is permitted
    /// with the Schur-complement factorization                       (OBSOLETE)
    ipc_ max_col;

    /// \brief
    /// an initial guess as to the integer workspace required by SBLS (OBSOLETE)
    ipc_ indmin;

    /// \brief
    /// an initial guess as to the real workspace required by SBLS    (OBSOLETE)
    ipc_ valmin;

    /// \brief
    /// an initial guess as to the workspace required by ULS          (OBSOLETE)
    ipc_ len_ulsmin;

    /// \brief
    /// the maximum number of iterative refinements allowed           (OBSOLETE)
    ipc_ itref_max;

    /// \brief
    /// the maximum number of CG iterations allowed. If cg_maxit < 0,
    /// this number will be reset to the dimension of the system + 1
    ///
    ipc_ cg_maxit;

    /// \brief
    /// the preconditioner to be used for the CG. Possible values are
    /// \li 0  automatic
    /// \li 1  no preconditioner, i.e, the identity within full factorization
    /// \li 2  full factorization
    /// \li 3  band within full factorization
    /// \li 4  diagonal using the barrier terms within full factorization
    ///        (OBSOLETE)
    /// \li 5  optionally supplied diagonal, G = D
    ipc_ preconditioner;

    /// \brief
    /// the semi-bandwidth of a band preconditioner, if appropriate   (OBSOLETE)
    ipc_ semi_bandwidth;

    /// \brief
    /// how much has A changed since last problem solved:
    /// 0 = not changed, 1 = values changed, 2 = structure changed
    ipc_ new_a;

    /// \brief
    /// how much has H changed since last problem solved:
    /// 0 = not changed, 1 = values changed, 2 = structure changed
    ipc_ new_h;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    ipc_ sif_file_device;

    /// \brief
    /// the threshold pivot used by the matrix factorization.
    /// See the documentation for SBLS for details                    (OBSOLETE)
    rpc_ pivot_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization when finding the ba
    /// See the documentation for ULS for details                     (OBSOLETE)
    rpc_ pivot_tol_for_basis;

    /// \brief
    /// any pivots smaller than zero_pivot in absolute value will be regarded to
    /// zero when attempting to detect linearly dependent constraints (OBSOLETE)
    rpc_ zero_pivot;

    /// \brief
    /// the computed solution which gives at least inner_fraction_opt times the
    /// optimal value will be found                                   (OBSOLETE)
    rpc_ inner_fraction_opt;

    /// \brief
    /// an upper bound on the permitted step (-ve will be reset to an appropriat
    /// large value by eqp_solve)
    rpc_ radius;

    /// \brief
    /// diagonal preconditioners will have diagonals no smaller than
    /// min_diagonal (OBSOLETE)
    rpc_ min_diagonal;

    /// \brief
    /// if the constraints are believed to be rank defficient and the residual
    /// at a "typical" feasible point is larger than
    /// max( max_infeasibility_relative * norm A, max_infeasibility_absolute )
    /// the problem will be marked as infeasible
    rpc_ max_infeasibility_relative;
    /// see max_infeasibility_relative
    rpc_ max_infeasibility_absolute;

    /// \brief
    /// the computed solution is considered as an acceptable approximation to th
    /// minimizer of the problem if the gradient of the objective in the
    /// preconditioning(inverse) norm is less than
    /// max( inner_stop_relative * initial preconditioning(inverse)
    /// gradient norm, inner_stop_absolute )
    rpc_ inner_stop_relative;
    /// see inner_stop_relative
    rpc_ inner_stop_absolute;
    /// see inner_stop_relative
    rpc_ inner_stop_inter;

    /// \brief
    /// if .find_basis_by_transpose is true, implicit factorization precondition
    /// will be based on a basis of A found by examining A's transpose
    /// (OBSOLETE)
    bool find_basis_by_transpose;

    /// \brief
    /// if .remove_dependencies is true, the equality constraints will be
    /// preprocessed to remove any linear dependencies
    ///
    bool remove_dependencies;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if .generate_sif_file is .true. if a SIF file describing the current
    /// problem is to be generated
    bool generate_sif_file;

    /// \brief
    /// name of generated SIF file containing input problem
    char sif_file_name[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for FDC
    struct fdc_control_type fdc_control;

    /// \brief
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;

    /// \brief
    /// control parameters for GLTR
    struct gltr_control_type gltr_control;
};

/**
 * time derived type as a C struct
 */
struct eqp_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the CPU time spent detecting linear dependencies
    rpc_ find_dependent;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    rpc_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    rpc_ solve;
    /// see solve
    rpc_ solve_inter;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent detecting linear dependencies
    rpc_ clock_find_dependent;

    /// \brief
    /// the clock time spent factorizing the required matrices
    rpc_ clock_factorize;

    /// \brief
    /// the clock time spent computing the search direction
    rpc_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct eqp_inform_type {

    /// \brief
    /// return status. See EQP_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of conjugate gradient iterations required
    ipc_ cg_iter;
    /// see cg_iter
    ipc_ cg_iter_inter;

    /// \brief
    /// the total integer workspace required for the factorization
    int64_t factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int64_t factorization_real;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by QPB_solve
    rpc_ obj;

    /// \brief
    /// timings (see above)
    struct eqp_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// return information from GLTR
    struct gltr_inform_type gltr_inform;
};

// *-*-*-*-*-*-*-*-*-*-    E Q P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void eqp_initialize( void **data,
                     struct eqp_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see eqp_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    E Q P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void eqp_read_specfile( struct eqp_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNEQP.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/eqp.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see eqp_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    E Q P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void eqp_import( struct eqp_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char H_type[],
                 ipc_ H_ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[],
                 const char A_type[],
                 ipc_ A_ne,
                 const ipc_ A_row[],
                 const ipc_ A_col[],
                 const ipc_ A_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see eqp_control_type)

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
  \li -3. The restrictions n > 0 or m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
       specified.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian, \f$H\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity',
   'zero' or 'none', the latter pair if \f$H=0\f$; lower or upper
   case variants are allowed.

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
   need not be set when the dense, diagonal or (scaled) identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

 @param[in]  A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the constraint Jacobian, \f$A\f$. It should be one of 'coordinate',
  'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

 @param[in]  A_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate storage scheme.
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


//  *-*-*-*-*-*-*-*-*-   E Q P _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void eqp_reset_control( struct eqp_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see eqp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-*-*-   E Q P _ S O L V E _ Q P   -*-*-*-*-*-*-*-*-*-*-*-*

void eqp_solve_qp( void **data,
                   ipc_ *status,
                   ipc_ n,
                   ipc_ m,
                   ipc_ h_ne,
                   const rpc_ H_val[],
                   const rpc_ g[],
                   const rpc_ f,
                   ipc_ a_ne,
                   const rpc_ A_val[],
                   rpc_ c[],
                   rpc_ x[],
                   rpc_ y[] );

/*!<
 Solve the quadratic program when the Hessian \f$H\f$ is available.

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
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.
  \li -7. The constraints appear to have no feasible point.
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
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
           specified.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

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

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the constraint Jacobian matrix
    \f$A\f$ in any of the available storage schemes.

 @param[in] c is a one-dimensional array of size m and type rpc_, that
    holds the linear term \f$c\f$  in the constraints.
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the
    linear constraints. The j-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

*/

//  *-*-*-*-*-*-*-*-*-*-   E Q P _ S O L V E _ S L D Q P  -*-*-*-*-*-*-*-*-*-

void eqp_solve_sldqp( void **data,
                      ipc_ *status,
                      ipc_ n,
                      ipc_ m,
                      const rpc_ w[],
                      const rpc_ x0[],
                      const rpc_ g[],
                      const rpc_ f,
                      ipc_ a_ne,
                      const rpc_ A_val[],
                      rpc_ c[],
                      rpc_ x[],
                      rpc_ y[] );

/*!<
 Solve the shifted least-distance quadratic program

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
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
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.
  \li -7. The constraints appear to have no feasible point.
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
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
           specified.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

  @param[in] w is a one-dimensional array of size n and type rpc_,
    that holds the values of the weights \f$w\f$.

  @param[in] x0 is a one-dimensional array of size n and type rpc_,
    that holds the values of the shifts \f$x^0\f$.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$.

@param[in] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the constraint Jacobian matrix
    \f$A\f$ in any of the available storage schemes.

 @param[in] c is a one-dimensional array of size m and type rpc_, that
    holds the linear term \f$c\f$ in the constraints.
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the
    linear constraints. The j-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

*/

//  *-*-*-*-*-*-*-*-*-*-   E Q P _ R E S O L V E _ Q P   -*-*-*-*-*-*-*-*-*-*-

void eqp_resolve_qp( void **data,
                     ipc_ *status,
                     ipc_ n,
                     ipc_ m,
                     const rpc_ g[],
                     const rpc_ f,
                     rpc_ c[],
                     rpc_ x[],
                     rpc_ y[] );

/*!<
 Resolve the quadratic program or shifted least-distance quadratic program
 when some or all of the data \f$g\f$, \f$f\f$ and \f$c\f$ has changed

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
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.
  \li -7. The constraints appear to have no feasible point.
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
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
           specified.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] c is a one-dimensional array of size m and type rpc_, that
    holds the linear term \f$c\f$ in the constraints.
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the
    linear constraints. The j-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

*/

// *-*-*-*-*-*-*-*-*-*-    E Q P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void eqp_information( void **data,
                      struct eqp_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see eqp_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    E Q P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void eqp_terminate( void **data,
                    struct eqp_control_type *control,
                    struct eqp_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see eqp_control_type)

  @param[out] inform   is a struct containing output information
              (see eqp_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example eqpt.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example eqptf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
