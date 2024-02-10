//* \file galahad_qpb.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_QPB C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package qpb

  \section qpb_intro Introduction

  \subsection qpb_purpose Purpose

  This package uses a primal-dual interior-point trust-region method
  to solve the <b>quadratic programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  subject to the general linear constraints
  \f[c_i^l  \leq  a_i^Tx  \leq c_i^u, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
   c_i^l \[<=] a_i^Tx \[<=] c_i^u, i = 1, ... , m,
  \n
\endmanonly
  and the simple bound constraints
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the \f$n\f$ by \f$n\f$ symmetric matrix \f$H\f$,
  the vectors \f$g\f$, \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$H\f$
  or the matrix \f$A\f$ of vectors \f$a_i\f$.

  If the matrix \f$H\f$ is positive semi-definite, a global
  solution is found. However, if \f$H\f$ is indefinite,
  the procedure may find a (weak second-order) critical point
  that is not the global solution to the given problem.

  \subsection qpb_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England, and
  Philippe L. Toint, University of Namur, Belgium.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection qpb_date Originally released

  December 1999, C interface January 2022.

  \subsection qpb_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[\mbox{(1a) $\hspace{66mm} A x = c\hspace{66mm}$}\f]
\manonly
  \n
  (1a) A x = c
  \n
\endmanonly
  and
  \f[\mbox{(1b) $\hspace{52mm} c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u,\hspace{52mm}$} \f]
\manonly
  \n
  (1b) c^l \[<=] c \[<=] c^u, x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[\mbox{(2a) $\hspace{58mm} H x + g = A^T y + z\hspace{58mm}$}\f]
\manonly
  \n
  (2a) H x + g = A^T y + z
  \n
\endmanonly
  where
  \f[\mbox{(2b) $\hspace{24mm} y = y^l + y^u, \;\; z = z^l + z^u, \,\,
   y^l \geq 0 , \;\;  y^u \leq 0 , \;\;
   z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,\hspace{24mm}$} \f]
\manonly
  \n
   (2b) y = y^l + y^u, z = z^l + z^u, y^l \[>=] 0, y^u \[<=] 0,
        z^l \[>=] 0 and z^u \[<=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[\mbox{(3) $\hspace{12mm}
  ( A x - c^l )^T y^l = 0  ,\;\;  ( A x - c^u )^T y^u = 0  ,\;\;
  (x -x^l )^T z^l = 0 \;\;  \mbox{and} \;\; (x -x^u )^T z^u = 0,\hspace{12mm} $}\f]
\manonly
  \n
  (3) (A x - c^l)^T y^l = 0, (A x - c^u)^T y^u = 0,
      (x -x^l)^T z^l = 0 and (x -x^u)^T z^u = 0,
  \n
\endmanonly
  where the vectors \f$y\f$ and \f$z\f$ are known as the Lagrange multipliers
  for2 the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection qpb_method Method

  Primal-dual interior point methods iterate towards a point
  that satisfies these conditions by ultimately aiming to satisfy
  (1a), (2a) and (3), while ensuring that (1b) and (2b) are
  satisfied as strict inequalities at each stage.  Appropriate norms of the
  amounts by  which (1a), (2a) and (3) fail to be satisfied are known as the
  primal and dual infeasibility, and the violation of complementary slackness,
  respectively. The fact that (1b) and (2b) are satisfied as strict
  inequalities gives such methods their other title, namely
  interior-point methods.

  The problem is solved in two phases. The goal of the first "initial
  feasible point" phase is to find a strictly interior point which is
  primal feasible, that is that {1a} is satisfied. The GALAHAD package
  LSQP is used for this purpose, and offers the options of either
  accepting the first strictly feasible point found, or preferably of
  aiming for the so-called "analytic center" of the feasible region.
  Having found such a suitable initial feasible point, the second
  "optimality" phase ensures that \req{4.1a} remains satisfied while
  iterating to satisfy dual feasibility (2a) and complementary
  slackness (3).  The optimality phase proceeds by approximately
  minimizing a sequence of barrier functions
\latexonly
  \[\frac{1}{2} x^T H x + g^T x + f -
   \mu \left[ \sum_{i=1}^{m} \log ( c_{i}  -  c_{i}^{l} )
   + \sum_{i=1}^{m} \log ( c_{i}^{u}  -  c_{i} )
   + \sum_{j=1}^{n} \log ( x_{j}  -  x_{j}^{l} )
   + \sum_{j=1}^{n} \log ( x_{j}^{u}  -  x_{j} ) \right],\]
\endlatexonly
\htmlonly
  $$\frac{1}{2} x^T H x + g^T x + f -
   \mu \left[ \sum_{i=1}^{m} \log ( c_{i}  -  c_{i}^{l} )
   + \sum_{i=1}^{m} \log ( c_{i}^{u}  -  c_{i} )
   + \sum_{j=1}^{n} \log ( x_{j}  -  x_{j}^{l} )
   + \sum_{j=1}^{n} \log ( x_{j}^{u}  -  x_{j} ) \right],$$
\endhtmlonly
\manonly
  \n
                  1/2 x^T H x + g^T x + f -
   mu [ sum_{i=1}^m log (c_i-c_i^l)  + sum_{i=1}^m log (c_i^u-c_i ) +
        sum_{j=1}^n log (x_j-x_j^l ) + sum_{j=1}^n log (x_j^u-x_j ) ]
  \n
\endmanonly
  for an approriate sequence of positive barrier parameters \f$\mu\f$
  converging to zero
  while ensuring that (1a) remain satisfied and that
  \f$x\f$ and \f$c\f$ are strictly interior points for (1b).
  Note that terms in the above sumations corresponding to infinite bounds are
  ignored, and that equality constraints are treated specially.

  Each of the barrier subproblems is solved using a trust-region method.
  Such a method generates a trial correction step \f$\Delta (x, c)\f$
  to the current iterate \f$(x, c)\f$ by replacing the nonlinear
  barrier function locally by a suitable quadratic model, and
  approximately minimizing this model in the intersection of \req{4.1a}
  and a trust region \f$\|\Delta (x, c)\| \leq \Delta\f$ for some
  appropriate strictly positive trust-region radius \f$\Delta\f$ and norm
  \f$\| \cdot \|\f$.  The step is accepted/rejected and the radius adjusted
  on the basis of how accurately the model reproduces the value of
  barrier function at the trial step. If the step proves to be
  unacceptable, a linesearch is performed along the step to obtain an
  acceptable new iterate. In practice, the natural primal "Newton" model
  of the barrier function is frequently less successful than an
  alternative primal-dual model, and consequently the primal-dual model
  is usually to be preferred.

  Once a barrier subproblem has been solved, extrapolation based on
  values and derivatives encountered on the central path is optionally
  used to determine a good starting point for the next subproblem.
  Traditional Taylor-series extrapolation has been superceded by more
  accurate Puiseux-series methods as these are particularly suited to
  deal with degeneracy.

  The trust-region subproblem is approximately solved using the combined
  conjugate-gradient/Lanczos method implemented in the GALAHAD package
  GLTR.  Such a method requires a suitable preconditioner, and in our
  case, the only flexibility we have is in approximating the model of
  the Hessian. Although using a fixed form of preconditioning is
  sometimes effective, we have provided the option of an automatic
  choice, that aims to balance the cost of applying the preconditioner
  against the needs for an accurate solution of the trust-region
  subproblem.  The preconditioner is applied using the GALAHAD matrix
  factorization package SBLS, but options at this stage are to factorize
  the preconditioner as a whole (the so-called "augmented system"
  approach), or to perform a block elimination first (the
  "Schur-complement" approach). The latter is usually to be prefered
  when a (non-singular) diagonal preconditioner is used, but may be
  inefficient if any of the columns of \f$A\f$ is too dense.

  In order to make the solution as efficient as possible, the variables
  and constraints are reordered internally by the GALAHAD package QPP
  prior to solution.  In particular, fixed variables, and free
  (unbounded on both sides) constraints are temporarily removed.

  \subsection qpb_references Reference

  The basic algorithm is a generalisation of those of

  Y. Zhang (1994),
   On the convergence of a class of infeasible interior-point methods for the
   horizontal linear complementarity problem,
   SIAM J. Optimization 4(1) 208-227,

  with a number of enhancements described by

  A. R. Conn, N. I. M. Gould, D. Orban and Ph. L. Toint (1999).
  A primal-dual trust-region algorithm for minimizing a non-convex
  function subject to general inequality and linear equality constraints.
  Mathematical Programming <b>87</b> 215-249.

  \subsection qpb_call_order Call order

  To solve a given problem, functions from the qpb package must be called
  in the following order:

  - \link qpb_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link qpb_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link qpb_import \endlink - set up problem data structures and fixed
      values
  - \link qpb_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link qpb_solve_qp \endlink - solve the quadratic program
  - \link qpb_information \endlink (optional) - recover information about
    the solution and solution process
  - \link qpb_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_QPB_H
#define GALAHAD_QPB_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_lsqp.h"
#include "galahad_fdc.h"
#include "galahad_sbls.h"
#include "galahad_gltr.h"
#include "galahad_fit.h"

/**
 * control derived type as a C struct
 */
struct qpb_control_type {

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
    /// any printing will start on this iteration
    ipc_ start_print;

    /// \brief
    /// any printing will stop on this iteration
    ipc_ stop_print;

    /// \brief
    /// at most maxit inner iterations are allowed
    ipc_ maxit;

    /// \brief
    /// the maximum number of iterative refinements allowed
    ipc_ itref_max;

    /// \brief
    /// the maximum number of CG iterations allowed. If cg_maxit < 0,
    /// this number will be reset to the dimension of the system + 1
    ///
    ipc_ cg_maxit;

    /// \brief
    /// specifies the type of indicator function used. Pssible values are
    /// \li 1 primal indicator: constraint active <=> distance to nearest
    ///     bound <= .indicator_p_tol
    /// \li 2 primal-dual indicator: constraint active <=> distance to nearest
    ///     bound <= .indicator_tol_pd * size of corresponding multiplier
    /// \li 3 primal-dual indicator: constraint active <=> distance to nearest
    ///     bound <= .indicator_tol_tapia * distance to same bound at previous
    ///     iteration
    ipc_ indicator_type;

    /// \brief
    /// indicate whether and how much of the input problem
    /// should be restored on output. Possible values are
    /// \li 0 nothing restored
    /// \li 1 scalar and vector parameters
    /// \li 2 all parameters
    ipc_ restore_problem;

    /// \brief
    /// should extrapolation be used to track the central path? Possible values
    /// \li 0 never
    /// \li 1 after the final major iteration
    /// \li 2 at each major iteration
    ipc_ extrapolate;

    /// \brief
    /// the maximum number of previous path points to use when fitting the data
    ipc_ path_history;

    /// \brief
    /// the factorization to be used. Possible values are
    /// \li 0  automatic
    /// \li 1  Schur-complement factorization
    /// \li 2  augmented-system factorization
    ipc_ factor;

    /// \brief
    /// the maximum number of nonzeros in a column of A which is permitted
    /// with the Schur-complement factorization
    ipc_ max_col;

    /// \brief
    /// an initial guess as to the integer workspace required by SBLS
    ipc_ indmin;

    /// \brief
    /// an initial guess as to the real workspace required by SBLS
    ipc_ valmin;

    /// \brief
    /// the number of iterations for which the overall infeasibility
    /// of the problem is not reduced by at least a factor .reduce_infeas
    /// before the problem is flagged as infeasible (see reduce_infeas)
    ipc_ infeas_max;

    /// \brief
    /// the preconditioner to be used for the CG is defined by precon.
    /// Possible values are
    /// \li 0  automatic
    /// \li 1  no preconditioner, i.e, the identity within full factorization
    /// \li 2  full factorization
    /// \li 3  band within full factorization
    /// \li 4  diagonal using the barrier terms within full factorization
    ipc_ precon;

    /// \brief
    /// the semi-bandwidth of a band preconditioner, if appropriate
    ipc_ nsemib;

    /// \brief
    /// the maximum order of path derivative to use
    ///
    ipc_ path_derivatives;

    /// \brief
    /// the order of (Puiseux) series to fit to the path data:
    ///  <=0 to fit all data
    ipc_ fit_order;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
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
    /// the required accuracy for the complementarity
    rpc_ stop_c;

    /// \brief
    /// tolerances used to terminate the inner iteration (for given mu):
    /// dual feasibility <= MAX( theta_d * mu ** beta, 0.99 * stop_d )
    /// complementarity  <= MAX( theta_c * mu ** beta, 0.99 * stop_d )
    rpc_ theta_d;
    /// see theta_d
    rpc_ theta_c;
    /// see theta_d
    rpc_ beta;

    /// \brief
    /// initial primal variables will not be closer than prfeas from their bound
    rpc_ prfeas;

    /// \brief
    /// initial dual variables will not be closer than dufeas from their bounds
    ///
    rpc_ dufeas;

    /// \brief
    /// the initial value of the barrier parameter. If muzero is not positive,
    /// it will be reset to an appropriate value
    rpc_ muzero;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least a
    /// factor reduce_infeas over .infeas_max iterations, the problem is flagged
    /// as infeasible (see infeas_max)
    rpc_ reduce_infeas;

    /// \brief
    /// if the objective function value is smaller than obj_unbounded, it will
    /// be flagged as unbounded from below.
    rpc_ obj_unbounded;

    /// \brief
    /// the threshold pivot used by the matrix factorization.
    /// See the documentation for SBLS for details
    rpc_ pivot_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization when attempting to
    /// detect linearly dependent constraints.
    /// See the documentation for FDC for details
    rpc_ pivot_tol_for_dependencies;

    /// \brief
    /// any pivots smaller than zero_pivot in absolute value will be regarded to
    /// zero when attempting to detect linearly dependent constraints
    rpc_ zero_pivot;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer
    /// than identical_bounds_tol will be reset to the average of their values
    rpc_ identical_bounds_tol;

    /// \brief
    /// the search direction is considered as an acceptable approximation
    /// to the minimizer of the model if the gradient of the model in the
    /// preconditioning(inverse) norm is less than
    /// max( inner_stop_relative * initial preconditioning(inverse)
    /// gradient norm, inner_stop_absolute )
    rpc_ inner_stop_relative;
    /// see inner_stop_relative
    rpc_ inner_stop_absolute;

    /// \brief
    /// the initial trust-region radius
    rpc_ initial_radius;

    /// \brief
    /// start terminal extrapolation when mu reaches mu_min
    rpc_ mu_min;

    /// \brief
    /// a search direction which gives at least inner_fraction_opt times the
    /// optimal model decrease will be found
    rpc_ inner_fraction_opt;

    /// \brief
    /// if .indicator_type = 1, a constraint/bound will be
    /// deemed to be active <=> distance to nearest bound <= .indicator_p_tol
    rpc_ indicator_tol_p;

    /// \brief
    /// if .indicator_type = 2, a constraint/bound will be deemed to be active
    /// <=> distance to nearest bound
    /// <= .indicator_tol_pd * size of corresponding multiplier
    rpc_ indicator_tol_pd;

    /// \brief
    /// if .indicator_type = 3, a constraint/bound will be deemed to be active
    /// <=> distance to nearest bound
    /// <= .indicator_tol_tapia * distance to same bound at previous iteration
    rpc_ indicator_tol_tapia;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// the equality constraints will be preprocessed to remove any linear
    /// dependencies if true
    bool remove_dependencies;

    /// \brief
    /// any problem bound with the value zero will be treated as if it were a
    /// general value if true
    bool treat_zero_bounds_as_general;

    /// \brief
    /// if .center is true, the algorithm will use the analytic center
    /// of the feasible set as its initial feasible point. Otherwise, a
    /// feasible point as close as possible to the initial point will be used.
    /// We recommend using the analytic center
    bool center;

    /// \brief
    /// if .primal, is true, a primal barrier method will be used in  place of t
    /// primal-dual method
    bool primal;

    /// \brief
    /// If extrapolation is to be used, decide between Puiseux and Taylor series
    bool puiseux;

    /// \brief
    /// if .feasol is true, the final solution obtained will be perturbed so
    /// that variables close to their bounds are moved onto these bounds
    bool feasol;

    /// \brief
    /// if .array_syntax_worse_than_do_loop is true, f77-style do loops will be
    /// used rather than f90-style array syntax for vector operations
    bool array_syntax_worse_than_do_loop;

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
    /// control parameters for LSQP
    struct lsqp_control_type lsqp_control;

    /// \brief
    /// control parameters for FDC
    struct fdc_control_type fdc_control;

    /// \brief
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;

    /// \brief
    /// control parameters for GLTR
    struct gltr_control_type gltr_control;

    /// \brief
    /// control parameters for FIT
    struct fit_control_type fit_control;
};

/**
 * time derived type as a C struct
 */
struct qpb_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    rpc_ preprocess;

    /// \brief
    /// the CPU time spent detecting linear dependencies
    rpc_ find_dependent;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    rpc_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    rpc_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    rpc_ solve;

    /// \brief
    /// the total CPU time spent in the initial-point phase of the package
    rpc_ phase1_total;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    /// in the inital-point phase
    rpc_ phase1_analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices in the inital-point
    /// phase
    rpc_ phase1_factorize;

    /// \brief
    /// the CPU time spent computing the search direction in the inital-point ph
    rpc_ phase1_solve;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    rpc_ clock_preprocess;

    /// \brief
    /// the clock time spent detecting linear dependencies
    rpc_ clock_find_dependent;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
    rpc_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    rpc_ clock_factorize;

    /// \brief
    /// the clock time spent computing the search direction
    rpc_ clock_solve;

    /// \brief
    /// the total clock time spent in the initial-point phase of the package
    rpc_ clock_phase1_total;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
    /// in the inital-point phase
    rpc_ clock_phase1_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices in the inital-poi
    /// phase
    rpc_ clock_phase1_factorize;

    /// \brief
    /// the clock time spent computing the search direction in the inital-point
    rpc_ clock_phase1_solve;
};

/**
 * inform derived type as a C struct
 */
struct qpb_inform_type {

    /// \brief
    /// return status. See QPB_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations required
    ipc_ iter;

    /// \brief
    /// the total number of conjugate gradient iterations required
    ipc_ cg_iter;

    /// \brief
    /// the return status from the factorization
    ipc_ factorization_status;

    /// \brief
    /// the total integer workspace required for the factorization
    int64_t factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int64_t factorization_real;

    /// \brief
    /// the total number of factorizations performed
    ipc_ nfacts;

    /// \brief
    /// the total number of "wasted" function evaluations during the linesearch
    ipc_ nbacts;

    /// \brief
    /// the total number of factorizations which were modified to ensure that th
    /// matrix was an appropriate preconditioner
    ipc_ nmods;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by QPB_solve
    rpc_ obj;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    rpc_ non_negligible_pivot;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// timings (see above)
    struct qpb_time_type time;

    /// \brief
    /// inform parameters for LSQP
    struct lsqp_inform_type lsqp_inform;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// return information from GLTR
    struct gltr_inform_type gltr_inform;

    /// \brief
    /// return information from FIT
    struct fit_inform_type fit_inform;
};

// *-*-*-*-*-*-*-*-*-*-    Q P B  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void qpb_initialize( void **data,
                     struct qpb_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see qpb_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    Q P B  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void qpb_read_specfile( struct qpb_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNQPB.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/qpb.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see qpb_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    Q P B  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void qpb_import( struct qpb_control_type *control,
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
  paramters for the remaining prcedures (see qpb_control_type)

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


//  *-*-*-*-*-*-*-*-*-   Q P B _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void qpb_reset_control( struct qpb_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see qpb_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-*-*-   Q P B _ S O L V E _ Q P   -*-*-*-*-*-*-*-*-*-*-*-*

void qpb_solve_qp( void **data,
                   ipc_ *status,
                   ipc_ n,
                   ipc_ m,
                   ipc_ h_ne,
                   const rpc_ H_val[],
                   const rpc_ g[],
                   const rpc_ f,
                   ipc_ a_ne,
                   const rpc_ A_val[],
                   const rpc_ c_l[],
                   const rpc_ c_u[],
                   const rpc_ x_l[],
                   const rpc_ x_u[],
                   rpc_ x[],
                   rpc_ c[],
                   rpc_ y[],
                   rpc_ z[],
                   ipc_ x_stat[],
                   ipc_ c_stat[] );

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
  \li -5. The simple-bound constraints are inconsistent.
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

 @param[in] c_l is a one-dimensional array of size m and type rpc_, that
    holds the lower bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_l, i = 0, ... ,  m-1, contains  \f$c^l_i\f$.

 @param[in] c_u is a one-dimensional array of size m and type rpc_, that
    holds the upper bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_u, i = 0, ... ,  m-1, contains  \f$c^u_i\f$.

 @param[in] x_l is a one-dimensional array of size n and type rpc_, that
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_, that
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] c is a one-dimensional array of size m and type rpc_, that
    holds the residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the general
    linear constraints. The j-th component
    of y, j = 0, ... , n-1, contains \f$y_j\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the optimal status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^Tx\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies  between its bounds.
*/

// *-*-*-*-*-*-*-*-*-*-    Q P B  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void qpb_information( void **data,
                      struct qpb_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see qpb_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    Q P B  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void qpb_terminate( void **data,
                    struct qpb_control_type *control,
                    struct qpb_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see qpb_control_type)

  @param[out] inform   is a struct containing output information
              (see qpb_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example qpbt.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example qpbtf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
