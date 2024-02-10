//* \file galahad_bllsb.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BLLSB C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. September 3rd 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package bllsb

  \section bllsb_intro Introduction

  \subsection bllsb_purpose Purpose

  This package uses a primal-dual interior-point crossover method
  to solve the <b>constrained linear least-squares problem</b>
  \f[\mbox{minimize}\;\; f(x) = \frac{1}{2} \| A_o x - b \|_W^2
    + \frac{1}{2} \sigma \| x \|^2 \f]
\manonly
  \n
  minimize f(x) := 1/2 ||Ax-b||_W^2 + 1/2 sigma ||x||^2
  \n
  subject to the simple bound constraints
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the norm \f$\|r\|_W = \sqrt{ \sum_{i=1}^o w_i r_i^2}\f$,
  the \f$o\f$ by \f$n\f$ design matrix \f$A_o\f$,
  the vectors \f$b\f$, \f$x^l\f$,
  \f$x^u\f$, the diagonal weights $w_i$ and the scalar \f$\sigma\f$ are given.
  Any of the constraint bounds \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$A_o\f$.

  \subsection bllsb_authors Authors

  N. I. M. Gould and D. P. Robinson, STFC-Rutherford Appleton Laboratory,
  England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection bllsb_date Originally released

  October 2022.

  \subsection bllsb_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[\mbox{(1) $\hspace{52mm} x^l \leq x \leq x^u,\hspace{52mm}$} \f]
\manonly
  \n
  (1) x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[\mbox{(2a) $\hspace{3mm} A_o^T W ( A_o x-b ) + \sigma x = z\f]
\manonly
  \n
  (2a) A_o^T W ( A_o x - b ) + \sigma x = z
  \n
\endmanonly
  where
  \f[\mbox{(2b) $\hspace{24mm} z = z^l + z^u, \,\,
   z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,\hspace{24mm}$} \f]
\manonly
  \n
   (2b) z = z^l + z^u, z^l \[>=] 0 and z^u \[<=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[\mbox{(3) $\hspace{12mm}
  (x -x^l )^T z^l = 0 \;\;  \mbox{and} \;\; (x -x^u )^T z^u = 0,\hspace{12mm} $}\f]
\manonly
  \n
  (3) (x -x^l)^T z^l = 0 and (x -x^u)^T z^u = 0,
  \n
\endmanonly
  where the diagonal matrix \f$W\f$ has diagonal entries \f$w_j^2\f$,
  \f$j = 1, \ldots , o\f$, where the vector \f$z\f$ are known as
  the dual variables for the bounds,
  and where the vector inequalities hold component-wise.

  \subsection bllsb_method Method

  Primal-dual interior point methods iterate towards a point
  that satisfies these conditions by ultimately aiming to satisfy
  (2a) and (3), while ensuring that (1) and (2b) are
  satisfied as strict inequalities at each stage.  Appropriate norms of the
  amounts by  which (2a) and (3) fail to be satisfied are known as the
  primal and dual infeasibility, and the violation of complementary slackness,
  respectively. The fact that (1) and (2b) are satisfied as strict
  inequalities gives such methods their other title, namely
  interior-point methods.

  The method aims at each stage to reduce the
  overall violation of (2a) and (3),
  rather than reducing each of the terms individually. Given an estimate
  \f$v = (x, c, y, y^l, y^u, z, z^l, z^u)\f$
  of the primal-dual variables, a correction
  \f$\Delta v = \Delta (x, c, y, y^l, y^u z, z^l, z^u)\f$
  is obtained by solving a suitable linear system of Newton equations for the
  nonlinear systems (2a) and a parameterized ``residual
  trajectory'' perturbation of (3); residual trajectories
  proposed by Zhang (1994) and Zhao and Sun (1999) are possibilities.
  An improved estimate \f$v + \alpha \Delta v\f$
  is then used, where the step-size \f$\alpha\f$
  is chosen as close to 1.0 as possible while ensuring both that
  (1) and (2b) continue to hold and that the individual components
  which make up the complementary slackness
  (3) do not deviate too significantly
  from their average value. The parameter that controls the perturbation
  of (3) is ultimately driven to zero.

  The Newton equations are solved by applying the
  GALAHAD matrix factorization package SLS.

  Optionally, the problem may be pre-processed temporarily to eliminate
  dependent constraints using the GALAHAD package FDC. This may
  improve the performance of the subsequent iteration.

  \subsection bllsb_references Reference

  The basic algorithm is a generalisation of those of

  Y. Zhang (1994),
   On the convergence of a class of infeasible interior-point methods for the
   horizontal linear complementarity problem,
   SIAM J. Optimization 4(1) 208-227,

  and

  G. Zhao and J. Sun (1999).
  On the rate of local convergence of high-order infeasible path-following
  algorithms for the \f$P_\ast\f$ linear complementarity problems,
  Computational Optimization and Applications 14(1) 293-307,

  with many enhancements described by

  N. I. M. Gould, D. Orban and D. P. Robinson (2013).
  Trajectory-following methods for large-scale  degenerate convex quadratic
  programming,
  Mathematical Programming Computation 5(2) 113-142

  and tailored for a regularized linear least-squares objective.

  \subsection bllsb_call_order Call order

  To solve a given problem, functions from the bllsb package must be called
  in the following order:

  - \link bllsb_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link bllsb_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link bllsb_import \endlink - set up problem data structures and fixed
      values
  - \link bllsb_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link bllsb_solve_blls \endlink - solve the east-squares problem
  - \link bllsb_information \endlink (optional) - recover information about
    the solution and solution process
  - \link bllsb_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$o\f$ by \f$n\f$ objective designt matrix \f$A_o\f$
  may be presented and stored in a variety of convenient input formats.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense Dense storage format

  The matrix \f$A_o\f$ is stored as a compact dense matrix either by rows
  or by colums, that is, the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In the row case, component \f$n \ast i + j\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$. By contrast,
  in the column case, component \f$m \ast j + i\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A_o\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays A_row and
  A_col and real array A_val, respectively, while the number of nonzeros
  is recorded as A_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A_o\f$ the i-th component of the
  integer array A_ptr holds the position of the first entry in this row,
  while A_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array A_col, and real array A_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsubsection unsymmetric_matrix_column_wise Sparse column-wise storage format
  Once again only the nonzero entries are stored, but this time
  they are ordered so that those in column j appear directly before those
  in column j+1. For the j-th column of \f$A_o\f$ the j-th component of the
  integer array A_ptr holds the position of the first entry in this column,
  while A_ptr(m) holds the total number of entries.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values   \f$A_{ij}\f$
  of the  nonzero entries in the j-th column are stored in components
  l = A_ptr(j), \f$\ldots\f$, A_ptr(j+1)-1,  \f$0 \leq j \leq n-1\f$,
  of the integer array A_row, and real array A_val, respectively.
  As before, For sparse matrices, this scheme almost always requires less
  storage than the co-ordinate format.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_BLLSB_H
#define GALAHAD_BLLSB_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_fdc.h"
#include "galahad_sls.h"
#include "galahad_fit.h"
#include "galahad_roots.h"
#include "galahad_cro.h"
#include "galahad_rpd.h"

/**
 * control derived type as a C struct
 */
struct bllsb_control_type {

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
    /// \li \f$\leq\f$ 0 gives no output,
    /// \li  = 1 gives a one-line summary for every iteration,
    /// \li  = 2 gives a summary of the inner iteration for each iteration,
    /// \li \f$\geq\f$ 3 gives increasingly verbose (debugging) output
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
    /// the number of iterations for which the overall infeasibility
    /// of the problem is not reduced by at least a factor .reduce_infeas
    /// before the problem is flagged as infeasible (see reduce_infeas)
    ipc_ infeas_max;

    /// \brief
    /// the initial value of the barrier parameter will not be changed for the
    /// first muzero_fixed iterations
    ///
    ipc_ muzero_fixed;

    /// \brief
    /// indicate whether and how much of the input problem
    /// should be restored on output. Possible values are
    /// \li 0 nothing restored
    /// \li 1 scalar and vector parameters
    /// \li 2 all parameters
    ipc_ restore_problem;

    /// \brief
    /// specifies the type of indicator function used. Possible values are
    /// \li 1 primal indicator: a constraint is active if and only if
    ///     the distance to its nearest bound \f$\leq\f$ .indicator_p_tol
    /// \li 2 primal-dual indicator: a constraint is active if and only if
    ///      the distance to its nearest bound \f$\leq\f$
    ///      .indicator_tol_pd * size of corresponding multiplier
    /// \li 3 primal-dual indicator: a constraint is active if and only if
    ///      the distance to its nearest bound \f$\leq\f$
    ///      .indicator_tol_tapia * distance to same bound at previous iteration
    ipc_ indicator_type;

    /// \brief
    /// which residual trajectory should be used to aim from the current iterate
    /// to the solution. Possible values are
    /// \li 1 the Zhang linear residual trajectory
    /// \li 2 the Zhao-Sun quadratic residual trajectory
    /// \li 3 the Zhang arc ultimately switching to the Zhao-Sun residual
    ///       trajectory
    /// \li 4 the mixed linear-quadratic residual trajectory
    /// \li 5 the Zhang arc ultimately switching to the mixed linear-quadratic
    ///       residual trajectory
    ipc_ arc;

    /// \brief
    /// the order of (Taylor/Puiseux) series to fit to the path data
    ipc_ series_order;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    ipc_ sif_file_device;

    /// \brief
    /// specifies the unit number to write generated QPLIB file describing the
    /// current problem
    ipc_ qplib_file_device;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// the required absolute and relative accuracies for the primal
    /// infeasibility
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
    /// initial primal variables will not be closer than .prfeas from their
    /// bounds
    rpc_ prfeas;

    /// \brief
    /// initial dual variables will not be closer than .dufeas from their
    /// bounds
    rpc_ dufeas;

    /// \brief
    /// the initial value of the barrier parameter. If muzero is not positive,
    /// it will be reset to an appropriate value
    rpc_ muzero;

    /// \brief
    /// the weight attached to primal-dual infeasibility compared to complementa
    /// when assessing step acceptance
    rpc_ tau;

    /// \brief
    /// individual complementarities will not be allowed to be smaller than
    /// gamma_c times the average value
    rpc_ gamma_c;

    /// \brief
    /// the average complementarity will not be allowed to be smaller than
    /// gamma_f times the primal/dual infeasibility
    rpc_ gamma_f;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least
    /// a factor .reduce_infeas over .infeas_max iterations, the problem is
    /// flagged as infeasible (see infeas_max)
    rpc_ reduce_infeas;

    /// \brief
    /// any pair of constraint bounds \f$(c_l,c_u)\f$ or \f$(x_l,x_u)\f$ that
    /// are closer than .identical_bounds_tol will be reset to the average
    /// of their values
    rpc_ identical_bounds_tol;

    /// \brief
    /// start terminal extrapolation when mu reaches mu_pounce
    rpc_ mu_pounce;

    /// \brief
    /// if .indicator_type = 1, a constraint/bound will be
    /// deemed to be active if and only if the distance to its nearest
    /// bound \f$\leq\f$ .indicator_p_tol
    rpc_ indicator_tol_p;

    /// \brief
    /// if .indicator_type = 2, a constraint/bound will be deemed to be active
    /// if and only if the distance to its nearest bound \f$\leq\f$
    /// .indicator_tol_pd * size of corresponding multiplier
    rpc_ indicator_tol_pd;

    /// \brief
    /// if .indicator_type = 3, a constraint/bound will be deemed to be active
    /// if and only if the distance to its nearest bound \f$\leq\f$
    /// .indicator_tol_tapia * distance to same bound at previous iteration
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
    /// if .just_feasible is true, the algorithm will stop as soon as a feasible
    /// point is found. Otherwise, the optimal solution to the problem will be
    /// found
    bool treat_separable_as_general;

    /// \brief
    /// if .treat_separable_as_general, is true, any separability in the
    /// problem structure will be ignored
    bool just_feasible;

    /// \brief
    /// if .getdua, is true, advanced initial values are obtained for the
    /// dual variables
    bool getdua;

    /// \brief
    /// decide between Puiseux and Taylor series approximations to the arc
    bool puiseux;

    /// \brief
    /// try every order of series up to series_order?
    bool every_order;

    /// \brief
    /// if .feasol is true, the final solution obtained will be perturbed so
    /// that variables close to their bounds are moved onto these bounds
    bool feasol;

    /// \brief
    /// if .balance_initial_complentarity is true, the initial complemetarity
    /// is required to be balanced
    ///
    bool balance_initial_complentarity;

    /// \brief
    ///
    /// if .crossover is true, cross over the solution to one defined by
    /// linearly-independent constraints if possible
    ///
    bool crossover;

    /// \brief
    /// if .reduced_pounce_system is true, eliminate fixed variables when
    /// solving the linear system required by the attempted pounce to
    /// the solution
    bool reduced_pounce_system;

    /// \brief
    /// if .space_critical is true, every effort will be made to use as little
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
    /// if .generate_qplib_file is .true. if a QPLIB file describing the current
    /// problem is to be generated
    bool generate_qplib_file;

 /// \brief
    /// the symmetric (indefinite) linear equation solver used
    char symmetric_linear_solver[31];

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
    /// control parameters for SLS used by BLLSB_pounce
    struct sls_control_type sls_pounce_control;

    /// \brief
    /// control parameters for FIT
    struct fit_control_type fit_control;

    /// \brief
    /// control parameters for ROOTS
    struct roots_control_type roots_control;

    /// \brief
    /// control parameters for CRO
    struct cro_control_type cro_control;
};

/**
 * time derived type as a C struct
 */
struct bllsb_time_type {

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
};

/**
 * inform derived type as a C struct
 */
struct bllsb_inform_type {

    /// \brief
    /// return status. See BLLSB_solve for details
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
    /// the number of threads used
    ipc_ threads;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by BLLSB_solve
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
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
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
    struct bllsb_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SLS
    struct sls_inform_type sls_inform;

    /// \brief
    /// inform parameters for SLS using BLLSB_pounce
    struct sls_inform_type sls_pounce_inform;

    /// \brief
    /// return information from FIT
    struct fit_inform_type fit_inform;

    /// \brief
    /// return information from ROOTS
    struct roots_inform_type roots_inform;

    /// \brief
    /// inform parameters for CRO
    struct cro_inform_type cro_inform;

    /// \brief
    /// inform parameters for RPD
    struct rpd_inform_type rpd_inform;
};

// *-*-*-*-*-*-*-*-*-*-    B L L S B  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void bllsb_initialize( void **data,
                      struct bllsb_control_type *control,
                      ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see bllsb_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    B L L S B  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void bllsb_read_specfile( struct bllsb_control_type *control,
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNBLLSB.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/bllsb.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see bllsb_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    B L L S B  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void bllsb_import( struct bllsb_control_type *control,
                  void **data,
                  ipc_ *status,
                  ipc_ n,
                  ipc_ o,
                  const char Ao_type[],
                  ipc_ Ao_ne,
                  const ipc_ Ao_row[],
                  const ipc_ Ao_col[],
                  ipc_ Ao_ptr_ne,
                  const ipc_ Ao_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bllsb_control_type)

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
  \li -3. The restrictions n > 0 or o > 0 or the requirement that a type
       contains its relevant string 'dense', 'dense_by_column', 'coordinate',
       'sparse_by_rows' or 'sparse_by_columns' has been violated.

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


//  *-*-*-*-*-*-*-*-   B L L S B _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void bllsb_reset_control( struct bllsb_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see bllsb_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-   B L L S B _ S O L V E _ B L L S   -*-*-*-*-*-*-*-*-*-

void bllsb_solve_blls( void **data,
                       ipc_ *status,
                       ipc_ n,
                       ipc_ o,
                       ipc_ Ao_ne,
                       const rpc_ Ao_val[],
                       const rpc_ b[],
                       rpc_ regularization_weight,
                       const rpc_ x_l[],
                       const rpc_ x_u[],
                       rpc_ x[],
                       rpc_ r[],
                       rpc_ z[],
                       ipc_ x_stat[],
                       rpc_ w[] );

/*!<
 Solve the constrained linear-least squares problem when the design matrix
  \f$A_o\f$ is available.

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

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] o is a scalar variable of type ipc_, that holds the number of
    residuals.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

  @param[in] ao_ne is a scalar variable of type ipc_, that holds the number of
    entries in the objective design matrix \f$A_o\f$.

  @param[in] Ao_val is a one-dimensional array of size ao_ne and type rpc_,
    that holds the values of the entries of the objective design
    matrix \f$A_o\f$ in any of the available storage schemes.

 @param[in] b is a one-dimensional array of size o and type rpc_, that
    holds the linear term \f$b\f$ of observations.
    The j-th component of b, i = 0, ... ,  o-1, contains  \f$b_i \f$.

 @param[in] regularization_weight is a scalar of type rpc_, that
    holds the non-negative regularization weight \f$\sigma \geq 0\f$.

 @param[in] x_l is a one-dimensional array of size n and type rpc_, that
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_, that
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] r is a one-dimensional array of size o and type rpc_, that
    holds the residual \f$r(x) = A_o x - b\f$.
    The i-th component of r, i = 0, ... ,  o-1, contains  \f$r_i(x) \f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[in] w is a one-dimensional array of size o and type rpc_, that
   holds the vector of strictly-positive observation weights \f$w\f$.
   If the weights are all one, w can be set to NULL.

*/


// *-*-*-*-*-*-*-*-*-*-    B L L S B  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void bllsb_information( void **data,
                       struct bllsb_inform_type *inform,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see bllsb_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    B L L S B  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void bllsb_terminate( void **data,
                     struct bllsb_control_type *control,
                     struct bllsb_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see bllsb_control_type)

  @param[out] inform   is a struct containing output information
              (see bllsb_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example bllsbt.c
   This is an example of how to use the package to solve a polyhedrally
   constrained, regularlized linear least-squares problem.
   A variety of supported objective and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example bllsbtf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
