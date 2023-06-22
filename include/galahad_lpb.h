//* \file galahad_lpb.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-21 AT 09:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LPB C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package lpb

  \section lpb_intro Introduction

  \subsection lpb_purpose Purpose

  This package uses a primal-dual interior-point method
  to solve the <b>linear programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = g^T x + f \f]
\manonly
  \n
  minimize q(x) := g^T x + f
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
  where the vectors \f$g\f$, \f$w\f$, \f$x^{0}\f$,
  \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix
  \f$A\f$ whose rows are the transposes of the vectors \f$a_i\f$.

  \subsection lpb_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory,
  England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection lpb_date Originally released

  August 2018, C interface September 2021.

  \subsection lpb_terminology Terminology

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
  \f[\mbox{(2a) $\hspace{3mm} g = A^T y + z$}\f]
\manonly
  \n
  (2a) g = A^T y + z
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
  where the vectors \f$y\f$ and \f$z\f$ are
  known as the Lagrange multipliers for
  the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection lpb_method Method

  Primal-dual interior point methods iterate towards a point
  that satisfies these conditions by ultimately aiming to satisfy
  (1a), (2a) and (3), while ensuring that (1b) and (2b) are
  satisfied as strict inequalities at each stage.  Appropriate norms of the
  amounts by  which (1a), (2a) and (3) fail to be satisfied are known as the
  primal and dual infeasibility, and the violation of complementary slackness,
  respectively. The fact that (1b) and (2b) are satisfied as strict
  inequalities gives such methods their other title, namely
  interior-point methods.

  The method aims at each stage to reduce the
  overall violation of (1a), (2a) and (3),
  rather than reducing each of the terms individually. Given an estimate
  \f$v = (x, c, y, y^l, y^u, z, z^l, z^u)\f$
  of the primal-dual variables, a correction
  \f$\Delta v = \Delta (x, c, y, y^l, y^u z, z^l, z^u)\f$
  is obtained by solving a suitable linear system of Newton equations for the
  nonlinear systems (1a), (2a) and a parameterized ``residual
  trajectory'' perturbation of (3); residual trajectories
  proposed by Zhang (1994) and Zhao and Sun (1999) are possibilities.
  An improved estimate \f$v + \alpha \Delta v\f$
  is then used, where the step-size \f$\alpha\f$
  is chosen as close to 1.0 as possible while ensuring both that
  (1b) and (2b) continue to hold and that the individual components
  which make up the complementary slackness
  (3) do not deviate too significantly
  from their average value. The parameter that controls the perturbation
  of (3) is ultimately driven to zero.

  The Newton equations are solved by applying the
  GALAHAD matrix factorization package SBLS, but there are options
  to factorize the matrix as a whole (the so-called "augmented system"
  approach), to perform a block elimination first (the "Schur-complement"
  approach), or to let the method itself decide which of the two
  previous options is more appropriate.
  The "Schur-complement" approach is usually to be preferred when all the
  weights are nonzero or when every variable is bounded (at least one side),
  but may be inefficient if any of the columns of \f$A\f$ is too dense.

  Optionally, the problem may be pre-processed temporarily to eliminate
  dependent constraints using the GALAHAD package FDC. This may
  improve the performance of the subsequent iteration.

  \subsection lpb_references Reference

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
  Mathematical Programming Computation 5(2) 113-142.

  \subsection lpb_call_order Call order

  To solve a given problem, functions from the lpb package must be called
  in the following order:

  - \link lpb_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link lpb_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link lpb_import \endlink - set up problem data structures and fixed
      values
  - \link lpb_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link lpb_solve_lp \endlink - solve the linear program
  - \link lpb_information \endlink (optional) - recover information about
    the solution and solution process
  - \link lpb_terminate \endlink - deallocate data structures

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
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_LPB_H
#define GALAHAD_LPB_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_fdc.h"
#include "galahad_sbls.h"
#include "galahad_fit.h"
#include "galahad_roots.h"
#include "galahad_cro.h"
#include "galahad_rpd.h"

/**
 * control derived type as a C struct
 */
struct lpb_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief
    /// the level of output required is specified by print_level
    int print_level;

    /// \brief
    /// any printing will start on this iteration
    int start_print;

    /// \brief
    /// any printing will stop on this iteration
    int stop_print;

    /// \brief
    /// at most maxit inner iterations are allowed
    int maxit;

    /// \brief
    /// the number of iterations for which the overall infeasibility
    /// of the problem is not reduced by at least a factor .reduce_infeas
    /// before the problem is flagged as infeasible (see reduce_infeas)
    int infeas_max;

    /// \brief
    /// the initial value of the barrier parameter will not be changed for the
    /// first muzero_fixed iterations
    ///
    int muzero_fixed;

    /// \brief
    /// indicate whether and how much of the input problem
    /// should be restored on output. Possible values are
    /// \li 0 nothing restored
    /// \li 1 scalar and vector parameters
    /// \li 2 all parameters
    int restore_problem;

    /// \brief
    /// specifies the type of indicator function used. Pssible values are
    /// \li 1 primal indicator: constraint active if and only if distance
    ///        to nearest bound <= .indicator_p_tol
    /// \li 2 primal-dual indicator: constraint active if and only if distance
    ///      the nearest bound
    ///      <= .indicator_tol_pd * size of corresponding multiplier
    /// \li 3 primal-dual indicator: constraint active if and only if distance
    ///     to the nearest bound <= .indicator_tol_tapia * distance to same
    /// bound at previous iteration
    int indicator_type;

    /// \brief
    /// which residual trajectory should be used to aim from the current
    /// iteration to the solution
    /// \li 1 the Zhang linear residual trajectory
    /// \li 2 the Zhao-Sun quadratic residual trajectory
    /// \li 3 the Zhang arc ultimately switching to the Zhao-Sun
    ///      residual trajectory
    /// \li 4 the mixed linear-quadratic residual trajectory
    int arc;

    /// \brief
    /// the order of (Taylor/Puiseux) series to fit to the path data
    int series_order;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    int sif_file_device;

    /// \brief
    /// specifies the unit number to write generated QPLIB file describing the
    /// current problem
    int qplib_file_device;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    real_wp_ infinity;

    /// \brief
    /// the required absolute and relative accuracies for the primal 
    /// infeasibility
    real_wp_ stop_abs_p;
    /// see stop_abs_p
    real_wp_ stop_rel_p;

    /// \brief
    /// the required absolute and relative accuracies for the dual infeasibility
    real_wp_ stop_abs_d;
    /// see stop_abs_d
    real_wp_ stop_rel_d;

    /// \brief
    /// the required absolute and relative accuracies for the complementarity
    real_wp_ stop_abs_c;
    /// see stop_abs_c
    real_wp_ stop_rel_c;

    /// \brief
    /// initial primal variables will not be closer than prfeas from their bound
    real_wp_ prfeas;

    /// \brief
    /// initial dual variables will not be closer than dufeas from their bounds
    ///
    real_wp_ dufeas;

    /// \brief
    /// the initial value of the barrier parameter. If muzero is not positive,
    /// it will be reset to an appropriate value
    real_wp_ muzero;

    /// \brief
    /// the weight attached to primal-dual infeasibility compared to 
    /// complementarity when assessing step acceptance
    real_wp_ tau;

    /// \brief
    /// individual complementarities will not be allowed to be smaller than
    /// gamma_c times the average value
    real_wp_ gamma_c;

    /// \brief
    /// the average complementarity will not be allowed to be smaller than
    /// gamma_f times the primal/dual infeasibility
    real_wp_ gamma_f;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least a
    /// factor reduce_infeas over .infeas_max iterations, the problem is flagged
    /// as infeasible (see infeas_max)
    real_wp_ reduce_infeas;

    /// \brief
    /// if the objective function value is smaller than obj_unbounded, it will
    /// be flagged as unbounded from below.
    real_wp_ obj_unbounded;

    /// \brief
    /// if W=0 and the potential function value is smaller than
    /// potential_unbounded * number of one-sided bounds,
    /// the analytic center will be flagged as unbounded
    real_wp_ potential_unbounded;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer
    /// than identical_bounds_tol will be reset to the average of their values
    real_wp_ identical_bounds_tol;

    /// \brief
    /// start terminal extrapolation when mu reaches mu_lunge
    real_wp_ mu_lunge;

    /// \brief
    /// if .indicator_type = 1, a constraint/bound will be deemed to be active
    /// if and only if distance to nearest bound <= .indicator_p_tol
    real_wp_ indicator_tol_p;

    /// \brief
    /// if .indicator_type = 2, a constraint/bound will be deemed to be active
    /// if and only if distance to nearest bound
    /// <= .indicator_tol_pd * size of corresponding multiplier
    real_wp_ indicator_tol_pd;

    /// \brief
    /// if .indicator_type = 3, a constraint/bound will be deemed to be active
    /// if and only if distance to nearest bound
    /// <= .indicator_tol_tapia * distance to same bound at previous iteration
    real_wp_ indicator_tol_tapia;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

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
    /// if .feasol is true, the final solution obtained will be perturbed so tha
    /// variables close to their bounds are moved onto these bounds
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
    /// if .generate_qplib_file is .true. if a QPLIB file describing the current
    /// problem is to be generated
    bool generate_qplib_file;

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
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;

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
struct lpb_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_wp_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    real_wp_ preprocess;

    /// \brief
    /// the CPU time spent detecting linear dependencies
    real_wp_ find_dependent;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to
    /// factorization
    real_wp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_wp_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    real_wp_ solve;

    /// \brief
    /// the total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    real_wp_ clock_preprocess;

    /// \brief
    /// the clock time spent detecting linear dependencies
    real_wp_ clock_find_dependent;

    /// \brief
    /// the clock time spent analysing the required matrices prior to
    /// factorization
    real_wp_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    real_wp_ clock_factorize;

    /// \brief
    /// the clock time spent computing the search direction
    real_wp_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct lpb_inform_type {

    /// \brief
    /// return status. See LPB_solve for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations required
    int iter;

    /// \brief
    /// the return status from the factorization
    int factorization_status;

    /// \brief
    /// the total integer workspace required for the factorization
    int64_t factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int64_t factorization_real;

    /// \brief
    /// the total number of factorizations performed
    int nfacts;

    /// \brief
    /// the total number of "wasted" function evaluations during the linesearch
    int nbacts;

    /// \brief
    /// the number of threads used
    int threads;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by LPB_solve
    real_wp_ obj;

    /// \brief
    /// the value of the primal infeasibility
    real_wp_ primal_infeasibility;

    /// \brief
    /// the value of the dual infeasibility
    real_wp_ dual_infeasibility;

    /// \brief
    /// the value of the complementary slackness
    real_wp_ complementary_slackness;

    /// \brief
    /// these values at the initial point (needed bg GALAHAD_CLPB)
    real_wp_ init_primal_infeasibility;
    /// see init_primal_infeasibility
    real_wp_ init_dual_infeasibility;
    /// see init_primal_infeasibility
    real_wp_ init_complementary_slackness;

    /// \brief
    /// the value of the logarithmic potential function
    /// sum -log(distance to constraint boundary)
    real_wp_ potential;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting 
    /// linearly dependent constraints
    real_wp_ non_negligible_pivot;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// checkpoints(i) records the iteration at which the criticality measures
    /// first fall below \f$10^{-i-1}\f$, i = 0, ..., 15 (-1 means not achieved)
    int checkpointsIter[16];
    /// see checkpointsIter
    real_wp_ checkpointsTime[16];

    /// \brief
    /// timings (see above)
    struct lpb_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;

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

// *-*-*-*-*-*-*-*-*-*-    L P B  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void lpb_initialize( void **data,
                     struct lpb_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lpb_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
    \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L P B  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void lpb_read_specfile( struct lpb_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNLPB.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/lpb.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see lpb_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    L P B  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void lpb_import( struct lpb_control_type *control,
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
  paramters for the remaining prcedures (see lpb_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restrictions n > 0 or m > 0 or requirement that A_type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
        has been violated.

 @param[in] n is a scalar variable of type int, that holds the number of
    variables.

 @param[in] m is a scalar variable of type int, that holds the number of
    general linear constraints.

 @param[in]  A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the constraint Jacobian, \f$A\f$. It should be one of 'coordinate',
  'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

 @param[in]  A_ne is a scalar variable of type int, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes.

 @param[in]  A_row is a one-dimensional array of size A_ne and type int, that
   holds the row indices of \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes,
   and in this case can be NULL.

 @param[in]  A_col is a one-dimensional array of size A_ne and type int,
   that holds the column indices of \f$A\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  A_ptr is a one-dimensional array of size n+1 and type int,
   that holds the starting position of each row of \f$A\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.
*/


//  *-*-*-*-*-*-*-*-*-   L P B _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void lpb_reset_control( struct lpb_control_type *control,
                        void **data,
                        int *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lpb_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  -*-*-*-*-*-*-*-*-*-*-*-   L P B _ S O L V E _ L P  -*-*-*-*-*-*-*-*-*-*-*-*-

void lpb_solve_lp( void **data,
                   int *status,
                   int n,
                   int m,
                   const real_wp_ g[],
                   const real_wp_ f,
                   int a_ne,
                   const real_wp_ A_val[],
                   const real_wp_ c_l[],
                   const real_wp_ c_u[],
                   const real_wp_ x_l[],
                   const real_wp_ x_u[],
                   real_wp_ x[],
                   real_wp_ c[],
                   real_wp_ y[],
                   real_wp_ z[],
                   int x_stat[],
                   int c_stat[] );

/*!<
 Solve the linear program.

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
  \li -3. The restrictions n > 0 and m > 0 or requirement that A_type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
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

 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in] m is a scalar variable of type int, that holds the number of
    general linear constraints.

 @param[in] g is a one-dimensional array of size n and type double, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type double, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] a_ne is a scalar variable of type int, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of size a_ne and type double,
    that holds the values of the entries of the constraint Jacobian matrix
    \f$A\f$ in any of the available storage schemes.

 @param[in] c_l is a one-dimensional array of size m and type double, that
    holds the lower bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_l, i = 0, ... ,  m-1, contains  \f$c^l_i\f$.

 @param[in] c_u is a one-dimensional array of size m and type double, that
    holds the upper bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_u, i = 0, ... ,  m-1, contains  \f$c^u_i\f$.

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
    holds the residual \f$c(x)\f$.
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i(x) \f$.

 @param[in,out] y is a one-dimensional array of size n and type double, that
    holds the values \f$y\f$ of the Lagrange multipliers for the general
    linear constraints. The j-th component
    of y, j = 0, ... , m-1, contains \f$y_i\f$.

 @param[in,out] z is a one-dimensional array of size n and type double, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] x_stat is a one-dimensional array of size n and type int, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] c_stat is a one-dimensional array of size m and type int, that
    gives the optimal status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^T x\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies  between its bounds.
*/

// *-*-*-*-*-*-*-*-*-*-    L P B  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lpb_information( void **data,
                      struct lpb_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lpb_inform_type)

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L P B  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void lpb_terminate( void **data,
                    struct lpb_control_type *control,
                    struct lpb_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see lpb_control_type)

  @param[out] inform   is a struct containing output information
              (see lpb_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example lpbt.c
   This is an example of how to use the package to solve a linear program.
   A variety of supported constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example lpbtf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
