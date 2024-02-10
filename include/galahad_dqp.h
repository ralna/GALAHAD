//* \file galahad_dqp.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_DQP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. December 24th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package dqp

  \section dqp_intro Introduction

  \subsection dqp_purpose Purpose

  This package uses a dual gradient-projection interior-point method
  to solve the <b>strictly convex quadratic programming problem</b>
  \f[(0) \;\;\; \mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  (0)   minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  or the <b>shifted least-distance problem</b>
  \f[\mbox{minimize}\;\; \frac{1}{2} \sum_{j=1}^n w_j^2 ( x_j^{ } - x_j^0 )^2
   + g^T x + f \f]
\manonly
  \n
   minimize 1/2 \sum_{j=1}^n w_j^2 ( x_j - x_j^0 )^2  + g^T x + f
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
  \f[x_j^l  \leq  x_j^{ } \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the \f$n\f$ by \f$n\f$ symmetric, positive-definite matrix
  \f$H\f$, the vectors \f$g\f$, \f$w\f$, \f$x^0\f$,
  \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$H\f$
  or the matrix \f$A\f$ of vectors \f$a_i\f$.

  \subsection dqp_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection dqp_date Originally released

  August 2012, C interface December 2021.

  \subsection dqp_terminology Terminology

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
  \f[\mbox{(2a) $\hspace{3mm} H x + g = A^T y + z \;\; (\mbox{or}
\;\;  W^{2} (x -x^0) + g = A^T y + z \;\; \mbox{for the shifted-least-distance type objective})$}\f]
\manonly
  \n
  (2a) H x + g = A^T y + z
       (or W^2 (x -x^0) + g = A^T y + z
        for the shifted-least-distance type objective)
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
  where the diagonal matrix \f$W^2\f$ has diagonal entries \f$w_j^2\f$,
  \f$j = 1, \ldots , n\f$, where the vectors \f$y\f$ and \f$z\f$ are
  known as the Lagrange multipliers for
  the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection dqp_method Method

  Dual gradient-projection methods solve (0) by instead solving the
  dual quadratic program
\latexonly
\[\mbox{(4) $\arr{ll}{\mbox{minimize}\;\; q^D(y^l, y^u, z^l, z^u) = & \!\!\! \frac{1}{2} [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1} [ A^T ( y^l  + y^u ) + z^l + z^u ] \\  & -  [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1} g - ( c^{l T} y^l + c^{u T} y^u +  x^{l T} z^l + x^{u T} z^u) \\ \mbox{subject to} & ( y^l, z^l ) \geq 0 \;\;\mbox{and} \;\;   (y^u, z^u) \leq 0,}$}\]
\endlatexonly
\htmlonly
$$\mbox{(4) $\arr{ll}{\mbox{minimize}\;\; q^D(y^l, y^u, z^l, z^u) = & \!\!\! \frac{1}{2} [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1} [ A^T ( y^l  + y^u ) + z^l + z^u ] \\  & -  [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1} g - ( c^{l T} y^l + c^{u T} y^u +  x^{l T} z^l + x^{u T} z^u) \\ \mbox{subject to} & ( y^l, z^l ) \geq 0 \;\;\mbox{and} \;\;   (y^u, z^u) \leq 0,}$}$$
\endhtmlonly
\manonly
  \n
  (4) minimize q^D(y^l, y^u, z^l, z^u) =
     1/2 [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1}
      [ A^T ( y^l  + y^u ) + z^l + z^u ]
     - [ ( y^l  + y^u )^T A + ( z^l + z^u ]^T ) H^{-1} g
     - ( c^{l T} y^l + c^{u T} y^u +  x^{l T} z^l + x^{u T} z^u
     subject to ( y^l, z^l ) >= 0 and(y^u, z^u) <= 0,
  \n
\endmanonly
  and then recovering the required solution from the linear system
\f[\mbox{$H x = - g + A^T ( y^l  + y^u ) + z^l + z^u.$}\f]
\manonly
  \n
   H x = - g + A^T ( y^l  + y^u ) + z^l + z^u.
  \n
\endmanonly
  The dual problem (4) is solved by an accelerated gradient-projection
  method comprising of alternating phases in which (i) the current
  projected dual gradient is traced downhill (the 'arc search')
  as far as possible and (ii) the  dual variables that
  are currently on their bounds are temporarily fixed and the unconstrained
  minimizer of \f$q^D(y^l, y^u, z^l, z^u)\f$ with respect to the
  remaining variables is sought; the minimizer in the second phase may itself
  need to be projected back into the dual feasible region (either
  using a brute-force backtrack or a second arc search).

  Both phases require
  the solution of sparse systems of symmetric linear equations, and these
  are handled by the GALAHAD matrix factorization package SBLS
  or the GALAHAD conjugate-gradient package GLTR.
  The systems are commonly singular, and this leads to a requirement to
  find the Fredholm Alternative for the given matrix and its right-hand side.
  In the non-singular case, there is an option to update existing factorizations
  using the "Schur-complement" approach given by the GALAHAD package SCU.

  Optionally, the problem may be pre-processed temporarily to eliminate
  dependentconstraints using the GALAHAD package FDC. This may improve the
  performance of the subsequent iteration.


  \subsection dqp_references Reference

  The basic algorithm is described in

  N. I. M. Gould and D. P. Robinson,
  ``A dual gradient-projection method
  for large-scale strictly-convex quadratic problems'',
  Computational Optimization and Applications
  <b>67(1)</b> (2017) 1-38.

  \subsection dqp_call_order Call order

  To solve a given problem, functions from the dqp package must be called
  in the following order:

  - \link dqp_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link dqp_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link dqp_import \endlink - set up problem data structures and fixed
      values
  - \link dqp_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link dqp_solve_qp \endlink - solve the quadratic program
     - \link dqp_solve_sldqp \endlink - solve the shifted least-distance problem
  - \link dqp_information \endlink (optional) - recover information about
    the solution and solution process
  - \link dqp_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_DQP_H
#define GALAHAD_DQP_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_fdc.h"
#include "galahad_sls.h"
#include "galahad_sbls.h"
#include "galahad_gltr.h"
#include "galahad_scu.h"
#include "galahad_rpd.h"

/**
 * control derived type as a C struct
 */
struct dqp_control_type {

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
    /// printing will only occur every print_gap iterations
    ipc_ print_gap;

    /// \brief
    /// which starting point should be used for the dual problem
    /// \li -1 user supplied comparing primal vs dual variables
    /// \li 0 user supplied
    /// \li 1 minimize linearized dual
    /// \li 2 minimize simplified quadratic dual
    /// \li 3 all free (= all active primal costraints)
    /// \li 4 all fixed on bounds (= no active primal costraints)
    ipc_ dual_starting_point;

    /// \brief
    /// at most maxit inner iterations are allowed
    ipc_ maxit;

    /// \brief
    /// the maximum permitted size of the Schur complement before a
    /// refactorization is performed (used in the case where there is no
    /// Fredholm Alternative, 0 = refactor every iteration)
    ipc_ max_sc;

    /// \brief
    /// a subspace step will only be taken when the current Cauchy step has
    /// changed no more than than cauchy_only active constraints; the subspace
    /// step will always be taken if cauchy_only < 0
    ipc_ cauchy_only;

    /// \brief
    /// how many iterations are allowed per arc search (-ve = as many as require
    ipc_ arc_search_maxit;

    /// \brief
    /// how many CG iterations to perform per DQP iteration (-ve reverts to n+1)
    ipc_ cg_maxit;

    /// \brief
    /// once a potentially optimal subspace has been found, investigate it
    /// \li 0 as per an ordinary subspace
    /// \li 1 by increasing the maximum number of allowed CG iterations
    /// \li 2 by switching to a direct method
    ipc_ explore_optimal_subspace;

    /// \brief
    /// indicate whether and how much of the input problem
    /// should be restored on output. Possible values are
    /// \li 0 nothing restored
    /// \li 1 scalar and vector parameters
    /// \li 2 all parameters
    ipc_ restore_problem;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    ipc_ sif_file_device;

    /// \brief
    /// specifies the unit number to write generated QPLIB file describing the
    /// current problem
    ipc_ qplib_file_device;

    /// \brief
    /// the penalty weight, rho. The general constraints are not enforced
    /// explicitly, but instead included in the objective as a penalty term
    /// weighted by rho when rho > 0. If rho <= 0, the general constraints are
    /// explicit (that is, there is no penalty term in the objective function)
    rpc_ rho;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// the required absolute and relative accuracies for the primal
    /// infeasibilies
    rpc_ stop_abs_p;
    /// see stop_abs_p
    rpc_ stop_rel_p;

    /// \brief
    /// the required absolute and relative accuracies for the dual infeasibility
    rpc_ stop_abs_d;
    /// see stop_abs_d
    rpc_ stop_rel_d;

    /// \brief
    /// the required absolute and relative accuracies for the complementarity
    rpc_ stop_abs_c;
    /// see stop_abs_c
    rpc_ stop_rel_c;

    /// \brief
    /// the CG iteration will be stopped as soon as the current norm of the
    /// preconditioned gradient is smaller than
    /// max( stop_cg_relative * initial preconditioned gradient,
    ///      stop_cg_absolute )
    rpc_ stop_cg_relative;
    /// see stop_cg_relative
    rpc_ stop_cg_absolute;

    /// \brief
    /// threshold below which curvature is regarded as zero if CG is used
    rpc_ cg_zero_curvature;

    /// \brief
    /// maximum growth factor allowed without a refactorization
    rpc_ max_growth;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer
    /// than identical_bounds_tol will be reset to the average of their values
    rpc_ identical_bounds_tol;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// the initial penalty weight (for DLP only)
    rpc_ initial_perturbation;

    /// \brief
    /// the penalty weight reduction factor (for DLP only)
    rpc_ perturbation_reduction;

    /// \brief
    /// the final penalty weight (for DLP only)
    rpc_ final_perturbation;

    /// \brief
    /// are the factors of the optimal augmented matrix required?
    /// (for DLP only)
    bool factor_optimal_matrix;

    /// \brief
    /// the equality constraints will be preprocessed to remove any linear
    /// dependencies if true
    bool remove_dependencies;

    /// \brief
    /// any problem bound with the value zero will be treated as if it were a
    /// general value if true
    bool treat_zero_bounds_as_general;

    /// \brief
    /// if .exact_arc_search is true, an exact piecewise arc search will be
    /// performed. Otherwise an ineaxt search using a backtracing Armijo
    /// strategy will be employed
    bool exact_arc_search;

    /// \brief
    /// if .subspace_direct is true, the subspace step will be calculated
    /// using a direct (factorization) method, while if it is false, an
    /// iterative (conjugate-gradient) method will be used.
    bool subspace_direct;

    /// \brief
    /// if .subspace_alternate is true, the subspace step will alternate
    /// between a direct (factorization) method and an iterative
    /// (GLTR conjugate-gradient) method. This will override .subspace_direct
    bool subspace_alternate;

    /// \brief
    /// if .subspace_arc_search is true, a piecewise arc search will be
    /// performed along the subspace step. Otherwise the search will stop
    /// at the firstconstraint encountered
    bool subspace_arc_search;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation
    /// error will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if .generate_sif_file is .true. if a SIF file describing the current
    /// problem is to be generated
    bool generate_sif_file;

    /// \brief
    /// if .generate_qplib_file is .true. if a QPLIB file describing the
    /// current problem is to be generated
    bool generate_qplib_file;

    /// \brief
    /// indefinite linear equation solver set in symmetric_linear_solver
    char symmetric_linear_solver[31];

    /// \brief
    /// definite linear equation solver
    char definite_linear_solver[31];

    /// \brief
    /// unsymmetric linear equation solver
    char unsymmetric_linear_solver[31];

    /// \brief
    /// name of generated SIF file containing input problem
    char sif_file_name[31];

    /// \brief
    /// name of generated QPLIB file containing input problem
    char qplib_file_name[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for FDC
    struct fdc_control_type fdc_control;

    /// \brief
    /// control parameters for SLS
    struct sls_control_type sls_control;

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
struct dqp_time_type {

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
    /// the CPU time spent analysing the required matrices prior to
    /// factorization
    rpc_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    rpc_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    rpc_ solve;

    /// \brief
    /// the CPU time spent in the linesearch
    rpc_ search;

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
    /// the clock time spent analysing the required matrices prior to
    /// factorization
    rpc_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    rpc_ clock_factorize;

    /// \brief
    /// the clock time spent computing the search direction
    rpc_ clock_solve;

    /// \brief
    /// the clock time spent in the linesearch
    rpc_ clock_search;
};

/**
 * inform derived type as a C struct
 */
struct dqp_inform_type {

    /// \brief
    /// return status. See DQP_solve for details
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
    /// the total number of iterations required
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
    /// the number of threads used
    ipc_ threads;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by DQP_solve
    rpc_ obj;

    /// \brief
    /// the value of the primal infeasibility
    rpc_ primal_infeasibility;

    /// \brief
    /// the value of the dual infeasibility
    rpc_ dual_infeasibility;

    /// \brief
    /// the value of the complementary slackness
    rpc_ complementary_slackness;

    /// \brief
    /// the smallest pivot that was not judged to be zero when detecting
    /// linearly dependent constraints
    rpc_ non_negligible_pivot;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// checkpoints(i) records the iteration at which the criticality measures
    /// first fall below \f$10^{-i-1}\f$, i = 0, ..., 15 (-1 means not achieved)
    ipc_ checkpointsIter[16];
    /// see checkpointsIter
    rpc_ checkpointsTime[16];

    /// \brief
    /// timings (see above)
    struct dqp_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SLS
    struct sls_inform_type sls_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// return information from GLTR
    struct gltr_inform_type gltr_inform;

    /// \brief
    /// inform parameters for SCU
    ipc_ scu_status;
    /// see scu_status
    struct scu_inform_type scu_inform;

    /// \brief
    /// inform parameters for RPD
    struct rpd_inform_type rpd_inform;
};

// *-*-*-*-*-*-*-*-*-*-    D Q P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void dqp_initialize( void **data,
                     struct dqp_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see dqp_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
    \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    D Q P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void dqp_read_specfile( struct dqp_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNDQP.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/dqp.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see dqp_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    D Q P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void dqp_import( struct dqp_control_type *control,
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
  paramters for the remaining prcedures (see dqp_control_type)

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
       'diagonal', 'scaled_identity' or 'identity' has been violated.
  \li -23. An entry from the strict upper triangle of \f$H\f$ has been
       specified.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian, \f$H\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', or 'identity';
   lower or upper case variants are allowed.

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


//  *-*-*-*-*-*-*-*-*-   D Q P _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void dqp_reset_control( struct dqp_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see dqp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-*-*-   D Q P _ S O L V E _ Q P   -*-*-*-*-*-*-*-*-*-*-*-*

void dqp_solve_qp( void **data,
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

//  *-*-*-*-*-*-*-*-*-*-   D Q P _ S O L V E _ S L D Q P  -*-*-*-*-*-*-*-*-*-

void dqp_solve_sldqp( void **data,
                      ipc_ *status,
                      ipc_ n,
                      ipc_ m,
                      const rpc_ w[],
                      const rpc_ x0[],
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

 @param[in,out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the general
    linear constraints. The i-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

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

// *-*-*-*-*-*-*-*-*-*-    D Q P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void dqp_information( void **data,
                      struct dqp_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see dqp_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    D Q P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void dqp_terminate( void **data,
                    struct dqp_control_type *control,
                    struct dqp_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see dqp_control_type)

  @param[out] inform   is a struct containing output information
              (see dqp_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example dqpt.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example dqptf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
