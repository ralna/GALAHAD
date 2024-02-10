//* \file galahad_wcp.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_WCP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 13th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package wcp

  \section wcp_intro Introduction

  \subsection wcp_purpose Purpose

  This package uses a primal-dual interior-point method
  to <b>find a well-centered interior point</b> \f$x\f$ for a set of
  general linear constraints
  \f[\mbox{(1)} \;\; c_i^l \leq a_i^Tx \leq c_i^u, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
   (1) c_i^l \[<=] a_i^Tx \[<=] c_i^u, i = 1, ... , m,
  \n
\endmanonly
  and the simple bound constraints
  \f[\mbox{(2)} \;\; x_j^l \leq x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   (2) x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  where the vectors
  \f$a_{i}\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$ and \f$x^u\f$ are given.
  More specifically, if possible, the package finds a solution to the
  system of primal optimality equations
  \f[\mbox{(3)} \;\; A x = c,\f]
\manonly
  \n
  (3) A x = c,
  \n
\endmanonly
  dual optimality equations
  \f[\mbox{(4) $\hspace{3mm} g = A^T y + z, \;\; y = y^l + y^u, \;\mbox{and} \; z = z^l + z^u,$}\f]
\manonly
  \n
  (4) g = A^T y + z, y = y^l + y^u and z = z^l + z^u,
  \n
\endmanonly
  and perturbed complementary slackness equations
  \f[\mbox{(5)} \;\;
  ( c_i - c^l_i ) y^l_i = (\mu_c^l)_i \;\mbox{and}\;
  ( c_i - c_i^u ) y^u_i = (\mu_c^u)_i, \;\;\;
   i = 1, \ldots , m, \f]
\manonly
  \n
  (c_i - c^l_i) y^l_i = (mu_c^l)_i and
  (c_i - c_i^u) y^u_i = (mu_c^u)_i, i = 1,...,m,
  \n
\endmanonly
  and
  \f[\mbox{(6)} \;\;
  ((x_j - x^l_j ) z_j^l = (\mu_x^l)_j \;\mbox{and}\;
  ( x_j - x^u_j ) z_j^u = (\mu_x^u)_j, \;\;\;
   j = 1, \ldots , n, \f]
\manonly
  \n
  (x_j - c^l_j) z^l_j = (mu_x^l)_j and
  (x_j - x_j^u) z^u_j = (mu_x^u)_i, j = 1,...,n,
  \n
\endmanonly
  for which
\latexonly
  \[
\mbox{(7)} \;\; c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u, \;\;
y^l \geq 0 , \;\; y^u \leq 0 , \;\; z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0
  \]
\endlatexonly
\htmlonly
  $$
\mbox{(7)} \;\; c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u, \;\;
y^l \geq 0 , \;\; y^u \leq 0 , \;\; z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0
  $$
\endhtmlonly
\manonly
  \n
  (7) c^l \[<=] c \[<=] c^u, x^l \[<=] x \[<=] x^u,
      y^l \[>=] 0, y^u \[<=] 0, z^l \[>=] 0 and z^u \[<=] 0
  \n
\endmanonly
  Here \f$A\f$ is the matrix whose rows are the \f$a_i^T\f$, \f$i = 1,
  \ldots , m\f$, \f$\mu_c^l\f$, \f$\mu_c^u\f$, \f$\mu_x^l\f$ and
  \f$\mu_x^u\f$ are vectors of strictly positive {\em targets}, \f$g\f$
  is another given target vector, and \f$(y^l, y^u)\f$ and \f$(z^l,
  z^u)\f$ are dual variables for the linear constraints and simple
  bounds respectively; \f$c\f$ gives the constraint value \f$A x\f$.
  Since (5)-(7) normally imply that
\latexonly
  \[
\mbox{(8)} \;\; c^l < c < c^u, \;\; x^l < x < x^u, \;\;
y^l > 0 , \;\; y^u < 0 , \;\; z^l > 0 \;\; \mbox{and} \;\; z^u < 0
  \]
\endlatexonly
\htmlonly
  $$
\mbox{(8)} \;\; c^l < c < c^u, \;\; x^l < x < x^u, \;\;
y^l > 0 , \;\; y^u < 0 , \;\; z^l > 0 \;\; \mbox{and} \;\; z^u < 0
  $$
\endhtmlonly
\manonly
  \n
  (8) c^l < c < c^u, x^l <; x < x^u,
      y^l > 0, y^u < 0, z^l > 0 and z^u < 0
  \n
\endmanonly
  such a primal-dual point \f$(x, c, y^l, y^u, z^l, z^l)\f$
  may be used, for example, as a feasible starting point for primal-dual
  interior-point methods for solving the linear programming problem
  of minimizing \f$g^T x\f$ subject to (1) and (2).

  Full advantage is taken of any zero coefficients in the vectors
  \f$a_{i}\f$. Any of the constraint bounds \f$c_{i}^l\f$,
  \f$c_{i}^u\f$, \f$x_{j}^l\f$ and \f$x_{j}^u\f$ may be infinite.
  The package identifies infeasible problems, and problems for which
  there is no strict interior, that is one or more of (8)
  only holds as an equality for all feasible points.

  \subsection wcp_authors Authors

  C. Cartis and N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection wcp_date Originally released

  July 2006, C interface January 2022.

  \subsection wcp_terminology Terminology

  \subsection wcp_method Method

  The algorithm is iterative, and at each major iteration attempts
  to find a solution to the perturbed system (3), (4),
  \f[\mbox{(9)}\;\;
  ( c_i - c^l_i + (\theta_c^l)_i )
  ( y^l_i + (\theta_y^l)_i )
  = (\mu_c^l)_i \;\mbox{and}\;
  ( c_i - c_i^u - (\theta_c^u)_i )
  ( y^u_i - (\theta_y^u)_i )
  = (\mu_c^u)_i, \;\;\;
   i = 1, \ldots , m,\f]
\manonly
  \n
       ( c_i - c^l_i + (theta_c^l)_i ) ( y^l_i + (theta_y^l)_i )
  (9)     = (mu_c^l)_i and
       ( c_i - c_i^u - (theta_c^u)_i ) ( y^u_i - (theta_y^u)_i )
          = (mu_c^u)_i, i = 1,...,m
  \n
\endmanonly
  \f[\mbox{(10)}\;\;
  ( x_j - x^l_j + (\theta_x^l)_j )
  ( z^l_j + (\theta_z^l)_j )
  = (\mu_x^l)_j \;\mbox{and}\;
  ( x_j - x_j^u - (\theta_x^u)_j )
  ( z^u_j - (\theta_z^u)_j )
  = (\mu_x^u)_j, \;\;\;
   j = 1, \ldots , n,\f]
\manonly
  \n
       ( x_j - x^l_j + (\theta_x^l)_j ) ( z^l_j + (\theta_z^l)_j )
  (10)     = (\mu_x^l)_j and
       ( x_j - x_j^u - (\theta_x^u)_j ) ( z^u_j - (\theta_z^u)_j )
           = (\mu_x^u)_j, j = 1,...,n,
  \n
\endmanonly
  and
  \f[\mbox{(11)}\;\;
  c^l - \theta_c^l < c < c^u + \theta_c^u, \;\;
  x^l - \theta_x^l < x < x^u + \theta_x^u, \;\;
   y^l > - \theta_y^l , \;\;
   y^u < \theta_y^u , \;\;
   z^l > - \theta_z^l \;\; \mbox{and} \;\;
   z^u < \theta_z^u ,\f]
\manonly
  \n
  c^l - theta_c^l < c < c^u + theta_c^u,
  x^l - theta_x^l < x < x^u + theta_x^u,
  y^l > - theta_y^l, y^u < theta_y^u,
  z^l > - theta_z^l and z^u < theta_z^,
  \n
\endmanonly
  where the vectors of perturbations
  \f$\theta^l_c\f$, \f$\theta^u_c\f$, \f$\theta^l_x\f$, \f$\theta^u_x\f$,
  \f$\theta^l_x\f$, \f$\theta^u_x\f$, \f$\theta^l_y\f$, \f$\theta^u_y\f$,
  \f$\theta^l_z\f$ and \f$\theta^u_z\f$,
  are non-negative. Rather than solve (3)-(4) and (9)-(11) exactly,
  we instead seek a feasible point for the easier relaxation (3)-(4) and
  \f[\mbox{(12)}\;\;
  \begin{array}{rcccll}
  \gamma (\mu_c^l)_i & \leq &
  ( c_i - c^l_i + (\theta_c^l)_i ) ( y^l_i + (\theta_y^l)_i )
  & \leq & (\mu_c^l)_i / \gamma & \mbox{and}\; \\
  \gamma (\mu_c^u)_i & \leq &
  ( c_i - c_i^u - (\theta_c^u)_i ) ( y^u_i - (\theta_y^u)_i )
  & \leq & (\mu_c^u)_i, /\gamma &
  i = 1, \ldots , m, \;\mbox{and}\; \\
  \gamma (\mu_x^l)_j & \leq &
  ( x_j - x^l_j + (\theta_x^l)_j ) ( z^l_j + (\theta_z^l)_j )
  & \leq & (\mu_x^l)_j /\gamma & \mbox{and}\; \\
  \gamma (\mu_x^u)_j & \leq &
  ( x_j - x_j^u - (\theta_x^u)_j )
  ( z^u_j - (\theta_z^u)_j )
  & \leq & (\mu_x^u)_j /\gamma , &j = 1, \ldots , n,
  \end{array}\f]
\manonly
  \n
       gamma (mu_c^l)_i
          \[<=] ( c_i - c^l_i + (theta_c^l)_i ) ( y^l_i + (theta_y^l)_i )
          \[<=] (mu_c^l)_i / gamma and
       gamma (mu_c^u)_i
          \[<=] ( c_i - c_i^u - (theta_c^u)_i ) ( y^u_i - (theta_y^u)_i )
 (12)     \[<=] (mu_c^u)_i, /gamma i = 1,...,m, and
       gamma (mu_x^l)_j
          \[<=] ( x_j - x^l_j + (theta_x^l)_j ) ( z^l_j + (theta_z^l)_j )
          \[<=] (mu_x^l)_j /gamma and
       gamma (mu_x^u)_j
          \[<=] ( x_j - x_j^u - (theta_x^u)_j ) ( z^u_j - (theta_z^u)_j )
          \[<=] (mu_x^u)_j /gamma , j = 1,...,n,
  \n
\endmanonly
  for some \f$\gamma \in (0,1]\f$ which is allowed to be smaller than one
  if there is a nonzero perturbation.

  Given any solution to (3)-(4) and (12) satisfying (11),
  the perturbations are reduced (sometimes to zero) so as to ensure that the
  current solution is feasible for the next perturbed problem. Specifically,
  the perturbation \f$(\theta^l_c)_i\f$ for the constraint \f$c_i \geq c^l_i\f$
  is set to zero if \f$c_i\f$ is larger than some given parameter
  \f$\epsilon > 0\f$.
  If not, but \f$c_i\f$ is strictly positive, the perturbation will be
  reduced by a multiplier \f$\rho \in (0,1)\f$. Otherwise, the new perturbation
  will be set to \f$\xi (\theta^l_c)_i + ( 1 - \xi ) ( c_i^l - c_i )\f$
  for some factor \f$\xi \in (0,1)\f$. Identical rules are used to reduce the
  remaining primal and dual perturbations.
  The targets \f$\mu_c^l\f$, \f$\mu_c^u\f$, \f$\mu_x^l\f$ and \f$\mu_x^u\f$
  will also be increased by the factor \f$\beta \geq 1\f$ for those
  (primal and/or dual) variables with strictly
  positive perturbations so as to try to accelerate the convergence.

  Ultimately the intention is to drive all the perturbations to zero.
  It can be shown that if the original problem (3)-(6) and (8)
  has a solution, the perturbations will be zero after a finite number of major
  iterations. Equally, if there is no interior solution (8)
  the sets of (primal and dual) variables that are necessarily at (one of) their
  bounds for all feasible points---we refer to these as {\em implicit}
  equalities---will be identified, as will the possibility that there is
  no point (interior or otherwise) in the primal and/or dual feasible regions.

  Each major iteration requires the solution \f$u = (x,c,z^l,z^u,y^l,y^u)\f$
  of the nonlinear system (3), (4) and (9)-(11)
  for fixed perturbations, using a minor iteration. The minor iteration
  uses a stabilized (predictor-corrector) Newton method, in which the arc
\latexonly
  $u(\alpha) = u + \alpha \dot{u} + \alpha^2 \ddot{u}, \alpha \in [0,1],$
\endlatexonly
\htmlonly
  $$u(\alpha) = u + \alpha \acute{u} + \alpha^2 \ddot{u}, \alpha \in [0,1],$$
\endhtmlonly
\manonly
u(alpha) = u + alpha u' + alpha^2 u'', alpha in [0,1], \
\endmanonly
involving the standard Newton step
\latexonly
$\dot{u}$
\endlatexonly
\htmlonly
&uacute;
\endhtmlonly
\manonly
u' \
\endmanonly
for the equations (3), (4), (9) and (10), optionally augmented by a corrector
\latexonly
$\ddot{u}$
\endlatexonly
\htmlonly
&uuml;
\endhtmlonly
\manonly
u'' \
\endmanonly
account for the nonlinearity in (9) and (10), is truncated so as to
ensure that
  \f[(c_i(\alpha) - c^l_i + (\theta_c^l)_i) (y^l_i(\alpha) + (\theta_y^l)_i)
  \geq \tau (\mu_c^l)_i \;\mbox{and}\;
  (c_i(\alpha) - c_i^u - (\theta_c^u)_i) (y^u_i(\alpha) - (\theta_y^u)_i)
  \geq \tau (\mu_c^u)_i, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
  (c_i(alpha) - c^l_i + (theta_c^l)_i) (y^l_i(alpha) + (theta_z^l)_i)
      \[>=] tau (mu_c^l)_i and
  (c_i(alpha) - c_i^u - (theta_c^u)_i ) (y^u_i(alpha) - (theta_z^u)_i)
      \[>=] tau (mu_c^u)_i, i = 1,...,m
  \n
\endmanonly
  and
  \f[(x_j(\alpha) - x^l_j + (\theta_x^l)_j) (z^l_j(\alpha) + (\theta_z^l)_j)
  \geq \tau (\mu_x^l)_j \;\mbox{and}\;
  (x_j(\alpha) - x_j^u - (\theta_x^u)_j ) (z^u_j(\alpha) - (\theta_z^u)_j)
  \geq \tau (\mu_x^u)_j, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
  (x_j(alpha) - x^l_j + (theta_x^l)_j) (z^l_j(alpha) + (theta_z^l)_j)
      \[>=] tau (mu_x^l)_j and
  (x_j(alpha) - x_j^u - (theta_x^u)_j ) (z^u_j(alpha) - (theta_z^u)_j)
      \[>=] tau (mu_x^u)_j, j = 1,...,n
  \n
\endmanonly
  for some \f$\tau \in (0,1)\f$, always holds, and also so that the norm
  of the residuals to (3), (4), (9) and (10)
  is reduced as much as possible.
  The Newton and corrector systems are solved using a factorization of
  the Jacobian of its defining functions (the so-called ``augmented system''
  approach) or of a reduced system in which some of the trivial equations are
  eliminated (the ``Schur-complement'' approach).
  The factors are obtained using the GALAHAD package SBLS.

  In order to make the solution as efficient as possible, the
  variables and constraints are reordered internally
  by the GALAHAD package QPP prior to solution.
  In particular, fixed variables, and
  free (unbounded on both sides) constraints are temporarily removed.
  In addition, an attempt to identify and remove linearly dependent
  equality constraints may be made by factorizing
\latexonly
  \[
  \mat{cc}{\alpha I & A^T_E \\ A_E & 0},
  \]
\endlatexonly
\htmlonly
$$
 \left( \begin{array}{cc} \alpha I & A^T_E \\ A_E & 0 \end{array}, \right)
$$
\endhtmlonly
\manonly
  \n
      ( alpha I   A_E^T ),
      (    A_E      0   )
  \n
\endmanonly
  where \f$A_E\f$ denotes the gradients of the equality constraints and
  \f$\alpha > 0\f$ is a given scaling factor,
  using the GALAHAD package SBLS, and examining small pivot blocks.

  \subsection wcp_references Reference

  The basic algorithm, its convergence analysis and results of
  numerical experiments are given in

  C. Cartis and N. I. M. Gould (2006).
  Finding a point n the relative interior of a polyhedron.
  Technical Report TR-2006-016, Rutherford Appleton Laboratory.

  \subsection wcp_call_order Call order

  To solve a given problem, functions from the wcp package must be called
  in the following order:

  - \link wcp_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link wcp_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link wcp_import \endlink - set up problem data structures and fixed
      values
  - \link wcp_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link wcp_find_wcp \endlink - find a well-centered point
  - \link wcp_information \endlink (optional) - recover information about
    the solution and solution process
  - \link wcp_terminate \endlink - deallocate data structures

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

  Both C-style (0 based) and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense Dense storage format
  The matrix \f$A\f$ is stored as a compact dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$ of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$, \f$0 \leq j \leq n-1\f$, are stored as
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
  \f$A_{ij}\f$ of the nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1, \f$0 \leq i \leq m-1\f$,
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
#ifndef GALAHAD_WCP_H
#define GALAHAD_WCP_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_fdc.h"
#include "galahad_sbls.h"

/**
 * control derived type as a C struct
 */
struct wcp_control_type {

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
    /// how to choose the initial point. Possible values are
    /// \li 0  the values input in X, shifted to be at least prfeas from
    ///     their nearest bound, will be used
    /// \li 1  the nearest point to the "bound average" 0.5(X_l+X_u) that
    ///     satisfies the linear constraints will be used
    ipc_ initial_point;

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
    /// the maximum number of iterative refinements allowed
    ipc_ itref_max;

    /// \brief
    /// the number of iterations for which the overall infeasibility of the
    /// problem is not reduced by at least a factor .required_infeas_reduction
    /// before the problem is flagged as infeasible (see required_infeas_reducti
    ipc_ infeas_max;

    /// \brief
    /// the strategy used to reduce relaxed constraint bounds.
    /// Possible values are
    /// \li 0 do not perturb the constraints
    /// \li 1 reduce all perturbations by the same amount with linear reduction
    /// \li 2 reduce each perturbation as much as possible with linear reduction
    /// \li 3 reduce all perturbations by the same amount with superlinear
    ///       reduction
    /// \li 4 reduce each perturbation as much as possible with superlinear
    ///       reduction
    ipc_ perturbation_strategy;

    /// \brief
    /// indicate whether and how much of the input problem should be restored
    /// on output. Possible values are
    /// \li 0 nothing restored
    /// \li 1 scalar and vector parameters
    /// \li 2 all parameters
    ipc_ restore_problem;

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
    /// initial primal variables will not be closer than prfeas from their bound
    rpc_ prfeas;

    /// \brief
    /// initial dual variables will not be closer than dufeas from their bounds
    ///
    rpc_ dufeas;

    /// \brief
    /// the target value of the barrier parameter. If mu_target is not positive,
    /// it will be reset to an appropriate value
    rpc_ mu_target;

    /// \brief
    /// the complemtary slackness x_i.z_i will be judged to lie within an
    /// acceptable margin around its target value mu as soon as
    /// mu_accept_fraction * mu <= x_i.z_i <= ( 1 / mu_accept_fraction ) * mu;
    /// the perturbations will be reduced as soon as all of the complemtary
    /// slacknesses x_i.z_i lie within acceptable bounds. mu_accept_fraction
    /// will be reset to ensure that it lies in the interval (0,1]
    rpc_ mu_accept_fraction;

    /// \brief
    /// the target value of the barrier parameter will be increased by
    /// mu_increase_factor for infeasible constraints every time the
    /// perturbations are adjusted
    ///
    rpc_ mu_increase_factor;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least
    /// a factor required_infeas_reduction over .infeas_max iterations, the
    /// problem is flagged as infeasible (see infeas_max)
    rpc_ required_infeas_reduction;

    /// \brief
    /// any primal or dual variable that is less feasible than implicit_tol will
    /// be regarded as defining an implicit constraint
    rpc_ implicit_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization.
    /// See the documentation for SBLS for details                    (obsolete)
    rpc_ pivot_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization when attempting to
    /// detect linearly dependent constraints.
    /// See the documentation for SBLS for details                    (obsolete)
    rpc_ pivot_tol_for_dependencies;

    /// \brief
    /// any pivots smaller than zero_pivot in absolute value will be regarded to
    /// zero when attempting to detect linearly dependent constraints (obsolete)
    rpc_ zero_pivot;

    /// \brief
    /// the constraint bounds will initially be relaxed by .perturb_start;
    /// this perturbation will subsequently be reduced to zero.
    /// If perturb_start < 0, the amount by which the bounds are relaxed will
    /// be computed automatically
    rpc_ perturb_start;

    /// \brief
    /// the test for rank defficiency will be to factorize
    /// ( alpha_scale I  A^T )
    /// (       A          0 )
    rpc_ alpha_scale;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha
    /// identical_bounds_tol will be reset to the average of their values
    rpc_ identical_bounds_tol;

    /// \brief
    /// the constraint perturbation will be reduced as follows:
    ///
    /// \li - if the variable lies outside a bound, the corresponding
    ///       perturbation
    /// will be reduced to reduce_perturb_factor * current pertubation
    /// + ( 1 - reduce_perturb_factor ) * violation
    /// \li - otherwise, if the variable lies within insufficiently_feasible
    /// of its bound the pertubation will be reduced to
    /// reduce_perturb_multiplier * current pertubation
    /// \li - otherwise if will be set to zero
    rpc_ reduce_perturb_factor;
    /// see reduce_perturb_factor
    rpc_ reduce_perturb_multiplier;
    /// see reduce_perturb_factor
    rpc_ insufficiently_feasible;

    /// \brief
    /// if the maximum constraint pertubation is smaller than
    /// perturbation_small and the violation is smaller than implicit_tol, the
    /// method will deduce that there is a feasible point but no interior
    rpc_ perturbation_small;

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
    bool just_feasible;

    /// \brief
    /// if .balance_initial_complementarity is .true. the initial complemetarity
    /// will be balanced
    bool balance_initial_complementarity;

    /// \brief
    /// if .use_corrector, a corrector step will be used
    bool use_corrector;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if .record_x_status is true, the array inform.X_status will be allocated
    /// and the status of the bound constraints will be reported on exit.
    bool record_x_status;

    /// \brief
    /// if .record_c_status is true, the array inform.C_status will be allocated
    /// and the status of the general constraints will be reported on exit.
    bool record_c_status;

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
};

/**
 * time derived type as a C struct
 */
struct wcp_time_type {

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
    ///  factorization
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
    ///  factorization
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
struct wcp_inform_type {

    /// \brief
    /// return status. See WCP_solve for details
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
    /// the number of general constraints that lie on (one) of their bounds for
    /// feasible solutions
    ipc_ c_implicit;

    /// \brief
    /// the number of variables that lie on (one) of their bounds for all
    /// feasible solutions
    ipc_ x_implicit;

    /// \brief
    /// the number of Lagrange multipliers for general constraints that lie on
    /// (one) of their bounds for all feasible solutions
    ipc_ y_implicit;

    /// \brief
    /// the number of dual variables that lie on (one) of their bounds for all
    /// feasible solutions
    ipc_ z_implicit;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by WCP_solve
    rpc_ obj;

    /// \brief
    /// the largest target value on termination
    rpc_ mu_final_target_max;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    rpc_ non_negligible_pivot;

    /// \brief
    /// is the returned primal-dual "solution" strictly feasible?
    bool feasible;

    /// \brief
    /// timings (see above)
    struct wcp_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    W C P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void wcp_initialize( void **data,
                     struct wcp_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see wcp_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    W C P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void wcp_read_specfile( struct wcp_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNWCP.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/wcp.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see wcp_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    W C P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void wcp_import( struct wcp_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char A_type[],
                 ipc_ A_ne,
                 const ipc_ A_row[],
                 const ipc_ A_col[],
                 const ipc_ A_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see wcp_control_type)

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

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

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


//  *-*-*-*-*-*-*-*-*-   W C P _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void wcp_reset_control( struct wcp_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see wcp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-*-*-   W C P _ F I N D _ W C P   -*-*-*-*-*-*-*-*-*-*-

void wcp_find_wcp( void **data,
                   ipc_ *status,
                   ipc_ n,
                   ipc_ m,
                   const rpc_ g[],
                   ipc_ a_ne,
                   const rpc_ A_val[],
                   const rpc_ c_l[],
                   const rpc_ c_u[],
                   const rpc_ x_l[],
                   const rpc_ x_u[],
                   rpc_ x[],
                   rpc_ c[],
                   rpc_ y_l[],
                   rpc_ y_u[],
                   rpc_ z_l[],
                   rpc_ z_u[],
                   ipc_ x_stat[],
                   ipc_ c_stat[] );

/*!<
 Find a well-centered point in the feasible region

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
  \li -4. The constraint bounds are inconsistent.
  \li -5. The constraints appear to have no feasible point.
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

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the target vector\f$g\f$.
    The j-th component of g, j = 0, ... ,  n-1, contains \f$g_j \f$.

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
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i(x) \f$.

 @param[in,out] y_l is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y^l\f$ of the Lagrange multipliers for the
    lower bounds on the general linear constraints. The j-th component
    of y_l, i = 0, ... , m-1, contains \f$y^l_i\f$.

 @param[in,out] y_u is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y^u\f$ of the Lagrange multipliers for the
    upper bounds on the general linear constraints. The j-th component
    of y_u, i = 0, ... , m-1, contains \f$y^u_i\f$.

 @param[in,out] z_l is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z^l\f$ of the dual variables for the lower bounds
    on the variables.
    The j-th component of z_l, j = 0, ... , n-1, contains \f$z^l_j\f$.

 @param[in,out] z_u is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z^u\f$ of the dual variables for the upper bounds
    on the variables.
    The j-th component of z_u, j = 0, ... , n-1, contains \f$z^u_j\f$.

 @param[out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the optimal status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^T x\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies  between its bounds.
*/

// *-*-*-*-*-*-*-*-*-*-    W C P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void wcp_information( void **data,
                      struct wcp_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information.

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see wcp_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    W C P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void wcp_terminate( void **data,
                    struct wcp_control_type *control,
                    struct wcp_inform_type *inform );

/*!<
  Deallocate all internal private storage.

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see wcp_control_type)

  @param[out] inform   is a struct containing output information
              (see wcp_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example wcpt.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example wcptf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
