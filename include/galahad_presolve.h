//* \file galahad_presolve.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_PRESOLVE C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. March 27th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package presolve

  \section presolve_intro Introduction

  \subsection presolve_purpose Purpose

  Presolving aims to <b>improve the formulation of a given optimization
    problem by applying a sequence of simple transformations</b>, and thereby
  to produce a \a reduced problem in a \a standard \a form that should be
  simpler to solve.  This reduced problem may then be passed to an
  appropriate solver.  Once the reduced problem has been solved, it is
  then \a restored to recover the solution for the original formulation.

  This package applies presolving techniques to a <b>linear</b>
  \f[\mbox{minimize}\;\; l(x) = g^T x + f \f]
\manonly
  \n
  minimize l(x) := g^T x + f
  \n
\endmanonly
  or <b>quadratic program</b>
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

  In addition, bounds on the Lagrange multipliers \f$y\f$ associated with
  the general linear constraints and on the dual variables \f$z\f$ associated
  with the simple bound constraints
  \f[ y_{i}^{l}  \leq  y_{i}  \leq  y_{i}^{u}, \;\;\;  i = 1, \ldots , m,\f]
\manonly
  \n
   y_j^i \[<=] y_i \[<=] y_i^u, i = 1, ... , m,
  \n
\endmanonly
  and
  \f[z_{i}^{l}  \leq  z_{i}  \leq  z_{i}^{u}, \;\;\;  i = 1, \ldots , n,\f]
\manonly
  \n
   z_j^l \[<=] z_j \[<=] z_j^u, j = 1, ... , n,
  \n
\endmanonly
  are also provided, where the \f$m\f$-dimensional vectors \f$y^l\f$ and
  \f$y^u\f$, as well as the \f$n\f$-dimensional vectors \f$x^l\f$ and \f$x^u\f$
  are given.  Any component of \f$c^l\f$, \f$c^u\f$, \f$x^l\f$, \f$x^u\f$,
  \f$y^l\f$, \f$y^u\f$, \f$z^l\f$ or \f$z^u\f$ may be infinite.

  \subsection presolve_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England
  and Ph. L. Toint, University of Namur, Belgium

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection presolve_date Originally released

  March 2002, C interface March 2022.

  \subsection presolve_terminology Terminology

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

  \subsection presolve_method Method

  The purpose of presolving is to exploit these equations in order to reduce
  the problem to the standard form defined as follows:
  - The variables are ordered so that their bounds appear in the order
    \f[\begin{array}{lccccc}
      \mbox{free}            &          &        & x &        &        \\
      \mbox{non-negativity}  &   0      & \leq & x &    &        \\
      \mbox{lower}           & x^l & \leq & x &   &        \\
      \mbox{range} & x^l & \leq & x & \leq & x^u\\
      \mbox{upper} &          &        & x & \leq & x^u \\
      \mbox{non-positivity}  &          &        & x & \leq &  0
    \end{array}\f]
\manonly
  \n
    free                     x
    non-negativity     0  <= x
    lower             x^l <= x
    range             x^l <= x  <= x^u
    upper                    x  <= x^u
    non-positivity           x  <=  0
  \n
\endmanonly
    Fixed variables are removed. Within each category, the variables
    are further ordered so that those with non-zero diagonal Hessian
    entries occur before the remainder.
  - The constraints are ordered so that their bounds appear in the order
    \f[\begin{array}{lccccc}
    \mbox{non-negativity} &     0    & \leq & A x &        &      \\
    \mbox{equality}       & c^l &   =  & A x &        &     \\
    \mbox{lower} & c^l & \leq & A x &        &     \\
    \mbox{range} & c^l & \leq & A x & \leq & c^u\\
    \mbox{upper} &          &        & A x & \leq & c^u \\
    \mbox{non-positivity} &          &        & A x & \leq & 0  \\
    \end{array}\f]
\manonly
  \n
    non-negativity     0  <= A x
    equality          c^l  = A x
    lower             c^l <= A x
    range             c^l <= A x <= c^u
    upper                    A x <= c^u
    non-positivity           A x <=  0
  \n
\endmanonly
    Free constraints are removed.
  - In addition, constraints may be removed or bounds tightened, to reduce the
  size of the feasible region or simplify the problem if this is possible, and
  bounds may be tightened on the dual variables and the multipliers
  associated  with the problem.

  The presolving algorithm proceeds by applying a (potentially long) series of
  simple transformations to the problem, each transformation introducing a
  further simplification of the problem. These involve the removal of empty and
  singleton rows, the removal of redundant and forcing primal constraints, the
  tightening of primal and dual bounds, the exploitation of linear singleton,
  linear doubleton and linearly unconstrained columns, the merging dependent
  variables, row sparsification and split equalities. Transformations are
  applied in successive passes, each pass involving the following actions:

  -# remove empty and singletons rows,
  -# try to eliminate variables that are linearly unconstrained,
  -# attempt to exploit the presence of linear singleton columns,
  -# attempt to exploit the presence of linear doubleton columns,
  -# complete the analysis of the dual constraints,
  -# remove empty and singletons rows,
  -# possibly remove dependent variables,
  -# analyze the primal constraints,
  -# try to make \f$A\f$ sparser by combining its rows,
  -# check the current status of the variables, dual variables and multipliers.

  All these transformations are applied to the structure of the original
  problem, which is only permuted to standard form after all transformations are
  completed. <em>Note that the Hessian and Jacobian of the resulting reduced
  problem are always stored in sparse row-wise format.</em> The reduced problem
  is then solved by a quadratic or linear programming solver, thus ensuring
  sufficiently small primal-dual feasibility and complementarity. Finally, the
  solution of the simplified problem is re-translated in the
  variables/constraints/format of the original problem formulation by a
  \a restoration phase.

  If the number of problem transformations exceeds
  \p control.transf_buffer_size,  the transformation buffer size,
  then they are saved in a ``history'' file, whose
  name may be chosen by specifying the control.transf_file_name control
  parameter,  When this is the case, this file
  is subsequently reread by \p presolve_restore_solution. It must not be
  altered by the user.

  Overall, the presolving process follows one of the two sequences:

  \f[\fbox{initialize} \rightarrow \left[ \fbox{apply transformations}
   \rightarrow \mbox{(solve problem)}
   \rightarrow \fbox{restore} \right] \rightarrow \fbox{terminate}\f]
  or
  \f[\fbox{initialize} \rightarrow \left[ \fbox{read specfile}
   \rightarrow \fbox{apply transformations}
   \rightarrow \mbox{(solve problem)}
   \rightarrow \fbox{restore} \right] \rightarrow \fbox{terminate}\f]
\manonly
 (ignore garbled doxygen phrase)
  \n
 --------------    [  -------------------------
 | initialize | -> [ | apply transformations | -> (solve problem) ->
 --------------    [  -------------------------
                      ----------- ]    -------------
                      | restore | ] -> | terminate |
                      ----------- ]    -------------
 or
 --------------    [ -----------------    -------------------------
 | initialize | -> [ | read specfile | -> | apply transformations | ->
 --------------    [ -----------------    -------------------------
                                         ----------- ]    -------------
                      (solve problem) -> | restore | ] -> | terminate |
                                         ----------- ]    -------------
  \n
\endmanonly

  where the procedure's control parameter may be modified by
  reading the specfile, and where (solve problem) indicates that the reduced
   problem is solved. Each of the
  ``boxed'' steps in these sequences corresponds to calling a specific
  routine of the package In the diagrams above, brackated subsequence of
  steps means that they can be repeated with problem having the same
  structure. The value of the \p problem.new_problem_structure
  must be true on entry of \p presolve_apply_to_problem on the
  first time it is used in this repeated subsequence. Such a subsequence must
  be terminated by a call to  \p presolve_terminate before presolving is
  applied to a problem with a different structure.

  Note that the values of the multipliers and dual variables (and thus of
  their respective bounds) depend on the functional form assumed for the
  Lagrangian function associated with the problem.  This form is given by
  \f[  L(x,y,z) = q x) - y\_{sign} * y^T (Ax-c) - z\_{sign} * z,\f]
  (considering only active constraints \f$A x = c\f$), where the parameters
  y_{sign} and z_{sign} are +1 or -1 and can be chosen by the user.
  Thus, if \f$y_{sign}\f$ = +1, the multipliers associated to active constraints
  originally posed as inequalities are non-negative if the inequality is a lower
  bound and non-positive if it is an upper bound. Obvioulsy they are not
  constrained in sign for constraints originally posed as equalities. These
  sign conventions are reversed if \f$y_{sign}\f$ = -1.
  Similarly, if \f$z_{sign}\f$ = +1}, the dual variables associated to active
  bounds are non-negative if the original bound is an lower bound, non-positive
  if it is an upper bound, or unconstrained in sign if the variables is fixed;
  and this convention is reversed in \f$z\_{sign}\f$ = -1}. The values of
  \f$z_{sign}\f$ and \f$y_{sign}\f$ may be chosen by setting the corresponding
  components of the \p control structure to \p 1 or \p -1.

  \subsection presolve_references Reference

  The algorithm is described in more detail in

  N. I. M. Gould and Ph. L. Toint (2004).
  Presolving for quadratic programming.
  Mathematical Programming <b>100</b>(1), pp 95--132.

  \subsection presolve_call_order Call order

  To solve a given problem, functions from the presolve package must be called
  in the following order:

  - \link presolve_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link presolve_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link presolve_import_problem \endlink - import the problem data and report
      the dimensions of the transformed problem
  - \link presolve_transform_problem \endlink - apply the presolve algorithm
      to transform the data
  - \link presolve_restore_solution \endlink - restore the solution from
         that of the transformed problem
  - \link presolve_information \endlink (optional) - recover information about
    the solution and solution process
  - \link presolve_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_PRESOLVE_H
#define GALAHAD_PRESOLVE_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct presolve_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// Determines the strategy for terminating the presolve
    /// analysis.  Possible values are:
    /// \li 1  presolving is continued as long as one of
    ///     the sizes of the problem (n, m, a_ne, or h_ne) is
    ///     being reduced;
    /// \li 2 presolving is continued as long as problem
    ///     transformations remain possible.
    /// NOTE: the maximum number of analysis passes
    /// (control.max_nbr_passes)  and the maximum number of
    /// problem transformations (control.max_nbr_transforms)
    /// set an upper limit on the  presolving effort
    /// irrespective of the choice of control.termination.
    /// The only effect of this latter parameter is to allow
    /// for early termination.
    ipc_ termination;

    /// \brief
    /// The maximum number of problem transformations, cumulated
    /// over all calls to \p presolve.
    ipc_ max_nbr_transforms;

    /// \brief
    /// The maximum number of analysis passes for problem analysis
    /// during a single call of \p presolve_transform_problem.
    ipc_ max_nbr_passes;

    /// \brief
    /// The relative accuracy at which the general linear
    /// constraints are satisfied at the exit of the solver.
    /// Note that this value is not used before the restoration
    /// of the problem.
    rpc_ c_accuracy;

    /// \brief
    /// The relative accuracy at which the dual feasibility
    /// constraints are satisfied at the exit of the solver.
    /// Note that this value is not used before the restoration
    /// of the problem.
    rpc_ z_accuracy;

    /// \brief
    /// The value beyond which a number is deemed equal to
    /// plus infinity
    /// (minus infinity being defined as its opposite)
    rpc_ infinity;

    /// \brief
    /// The unit number associated with the device used for
    /// printout.
    ipc_ out;

    /// \brief
    /// The unit number associated with the device used for
    /// error ouput.
    ipc_ errout;

    /// \brief
    /// The level of printout requested by the user. Can take
    /// the values:
    /// \li 0  no printout is produced
    /// \li 1 only reports the major steps in the analysis
    /// \li 2 reports the identity of each problem
    ///              transformation
    /// \li 3 reports more details
    /// \li 4 reports lots of information.
    /// \li 5 reports a completely silly amount of information
    ipc_ print_level;

    /// \brief
    /// true if dual transformations of the problem are allowed.
    /// Note that this implies that the reduced problem is solved
    /// accurately (for the dual feasibility condition to hold)
    /// as to be able to restore the problem to the original
    /// constraints and variables. false prevents dual
    /// transformations to be applied, thus allowing for inexact
    /// solution of the reduced problem. The setting of this control
    /// parameter overides that of get_z, get_z_bounds, get_y,
    /// get_y_bounds, dual_constraints_freq, singleton_columns_freq,
    /// doubleton_columns_freq, z_accuracy, check_dual_feasibility.
    bool dual_transformations;

    /// \brief
    /// true if the redundant variables and constraints (i.e.
    /// variables that do not appear in the objective
    /// function and appear with a consistent sign in the
    /// constraints) are to be removed with their associated
    /// constraints before other transformations are attempted.
    bool redundant_xc;

    /// \brief
    /// The frequency of primal constraints analysis in terms of
    /// presolving passes.  A value of j = 2 indicates that primal
    /// constraints are analyzed every 2 presolving passes. A zero
    /// value indicates that they are never analyzed.
    ipc_ primal_constraints_freq;

    /// \brief
    /// The frequency of dual constraints analysis in terms of
    /// presolving passes.  A value of j = 2 indicates that dual
    /// constraints are analyzed every 2 presolving passes.  A zero
    /// value indicates that they are never analyzed.
    ipc_ dual_constraints_freq;

    /// \brief
    /// The frequency of singleton column analysis in terms of
    /// presolving passes.  A value of j = 2 indicates that
    /// singleton columns are analyzed every 2 presolving passes.
    /// A zero value indicates that they are never analyzed.
    ipc_ singleton_columns_freq;

    /// \brief
    /// The frequency of doubleton column analysis in terms of
    /// presolving passes.  A value of j indicates that doubleton
    /// columns are analyzed every 2 presolving passes.  A zero
    /// value indicates that they are never analyzed.
    ipc_ doubleton_columns_freq;

    /// \brief
    /// The frequency of the attempts to fix linearly unconstrained
    /// variables, expressed in terms of presolving passes.  A
    /// value of j = 2 indicates that attempts are made every 2
    /// presolving passes.  A zero value indicates that no attempt
    /// is ever made.
    ipc_ unc_variables_freq;

    /// \brief
    /// The frequency of search for dependent variables in terms of
    /// presolving passes.  A value of j = 2 indicates that
    /// dependent variables are searched for every 2 presolving
    /// passes.  A zero value indicates that they are never
    /// searched for.
    ipc_ dependent_variables_freq;

    /// \brief
    /// The frequency of the attempts to make A sparser in terms of
    /// presolving passes.  A value of j = 2 indicates that attempts
    /// are made every 2 presolving passes.  A zero value indicates
    /// that no attempt is ever made.
    ipc_ sparsify_rows_freq;

    /// \brief
    /// The maximum percentage of fill in each row of A. Note that
    /// this is a row-wise measure: globally fill never exceeds
    /// the storage initially used for A, no matter how large
    /// control.max_fill is chosen. If max_fill is negative,
    /// no limit is put on row fill.
    ipc_ max_fill;

    /// \brief
    /// The unit number to be associated with the file(s) used
    /// for saving problem transformations on a disk file.
    ipc_ transf_file_nbr;

    /// \brief
    /// The number of transformations that can be kept in memory
    /// at once (that is without being saved on a disk file).
    ipc_ transf_buffer_size;

    /// \brief
    /// The exit status of the file where problem transformations
    /// are saved:
    /// \li 0 the file is not deleted after program termination
    /// \li 1 the file is not deleted after program termination
    ipc_ transf_file_status;

    /// \brief
    /// The name of the file (to be) used for storing
    /// problem transformation on disk.
    /// NOTE: this parameter must be identical for all calls to
    /// \p presolve following \p presolve_read_specfile. It can
    /// then only be changed after calling presolve_terminate.
    char transf_file_name[31];

    /// \brief
    /// Determines the convention of sign used for the multipliers
    /// associated with the general linear constraints.
    /// \li 1 All multipliers corresponding to active
    ///     inequality constraints are non-negative for
    ///     lower bound constraints and non-positive for
    ///     upper bounds constraints.
    /// \li -1 All multipliers corresponding to active
    ///     inequality constraints are non-positive for
    ///     lower bound constraints and non-negative for
    ///     upper bounds constraints.
    ipc_ y_sign;

    /// \brief
    /// Determines whether or not the multipliers corresponding
    /// to constraints that are inactive at the unreduced point corresponding
    /// to the reduced point on input to \p presolve_restore_solution
    /// must be set to zero. Possible values are:
    /// associated with the general linear constraints.
    /// \li 0 All multipliers corresponding to inactive
    ///     inequality constraints are forced to zero,
    ///     possibly at the expense of deteriorating the
    ///     dual feasibility condition.
    /// \li 1 Multipliers corresponding to inactive
    ///     inequality constraints are left unaltered.
    ipc_ inactive_y;

    /// \brief
    /// Determines the convention of sign used for the dual
    /// variables associated with the bound constraints.
    /// \li 1 All dual variables corresponding to
    ///     active lower bounds are non-negative, and
    ///     non-positive for active upper bounds.
    /// \li -1 All dual variables corresponding to
    ///     active lower bounds are non-positive, and
    ///     non-negative for active upper bounds.
    ipc_ z_sign;

    /// \brief
    /// Determines whether or not the dual variables corresponding
    /// to bounds that are inactive at the unreduced point corresponding
    /// to the reduced point on input to \p presolve_restore_solution
    /// must be set to zero. Possible values are:
    /// associated with the general linear constraints.
    /// \li 0: All dual variables corresponding to
    ///     inactive bounds are forced to zero,
    ///     possibly at the expense of deteriorating the
    ///     dual feasibility condition.
    /// \li 1 Dual variables corresponding to inactive
    ///     bounds are left unaltered.
    ipc_ inactive_z;

    /// \brief
    /// The type of final bounds on the variables returned by the
    /// package.  This parameter can take the values:
    /// \li 0 the final bounds are the tightest bounds
    ///     known on the variables (at the risk of
    ///     being redundant with other constraints,
    ///     which may cause degeneracy);
    /// \li 1 the best known bounds that are known to
    ///     be non-degenerate. This option implies
    ///     that an additional real workspace of size
    ///     2 * n must be allocated.
    /// \li 2 the loosest bounds that are known to
    ///     keep the problem equivalent to the
    ///     original problem. This option also
    ///     implies that an additional real
    ///     workspace of size 2 * n must be
    ///     allocated.
    ///
    /// NOTE: this parameter must be identical for all calls to
    /// presolve (except presolve_initialize).
    ipc_ final_x_bounds;

    /// \brief
    /// The type of final bounds on the dual variables returned by
    /// the package.  This parameter can take the values:
    /// \li 0 the final bounds are the tightest bounds
    ///     known on the dual variables (at the risk
    ///     of being redundant with other constraints,
    ///     which may cause degeneracy);
    /// \li 1 the best known bounds that are known to
    ///     be non-degenerate. This option implies
    ///     that an additional real workspace of size
    ///     2 * n must be allocated.
    /// \li 2 the loosest bounds that are known to
    ///     keep the problem equivalent to the
    ///     original problem. This option also
    ///     implies that an additional real
    ///     workspace of size 2 * n must be allocated.
    ///
    /// NOTE: this parameter must be identical for all calls to
    /// presolve (except presolve_initialize).
    ipc_ final_z_bounds;

    /// \brief
    /// The type of final bounds on the constraints returned by the
    /// package.  This parameter can take the values:
    /// \li 0 the final bounds are the tightest bounds
    ///     known on the constraints (at the risk of
    ///     being redundant with other constraints,
    ///     which may cause degeneracy);
    /// \li 1 the best known bounds that are known to
    ///     be non-degenerate. This option implies
    ///     that an additional real workspace of size
    ///     2 * m must be allocated.
    /// \li 2 the loosest bounds that are known to
    ///     keep the problem equivalent to the
    ///     original problem. This option also
    ///     implies that an additional real
    ///     workspace of size 2 * n must be
    ///     allocated.
    ///
    /// NOTES:
    /// 1) This parameter must be identical for all calls to
    /// presolve (except presolve_initialize).
    /// 2) If different from 0, its value must be identical
    /// to that of control.final_x_bounds.
    ipc_ final_c_bounds;

    /// \brief
    /// The type of final bounds on the multipliers returned by the
    /// package.  This parameter can take the values:
    /// \li 0 the final bounds are the tightest bounds
    ///     known on the multipliers (at the risk of
    ///     being redundant with other constraints,
    ///     which may cause degeneracy);
    /// \li 1 the best known bounds that are known to
    ///     be non-degenerate. This option implies
    ///     that an additional real workspace of size
    ///     2 * m must be allocated.
    /// \li 2 the loosest bounds that are known to
    ///     keep the problem equivalent to the
    ///     original problem. This option also
    ///     implies that an additional real
    ///     workspace of size 2 * n must be
    ///     allocated.
    ///
    /// NOTE: this parameter must be identical for all calls to
    /// presolve (except presolve_initialize).
    ipc_ final_y_bounds;

    /// \brief
    /// The level of feasibility check (on the values of x) at
    /// the start of the restoration phase.  This parameter can
    /// take the values:
    /// \li 0 no check at all;
    /// \li 1 the primal constraints are recomputed at x
    ///     and a message issued if the computed value
    ///     does not match the input value, or if it is
    ///     out of bounds (if control.print_level >= 2);
    /// \li 2 the same as for 1, but presolve is
    ///     terminated if an incompatibilty is detected.
    ipc_ check_primal_feasibility;

    /// \brief
    /// The level of dual feasibility check (on the values of x,
    /// y and z) at the start of the restoration phase.
    /// This parameter can take the values:
    /// \li 0 no check at all;
    /// \li 1 the dual feasibility condition is  recomputed
    ///    at ( x, y, z ) and a message issued if the
    ///    computed value does not match the input value
    ///    (if control.print_level >= 2);
    /// \li 2 the same as for 1, but presolve is
    ///     terminated if an incompatibilty is detected.
    ///     The last two values imply the allocation of an additional
    ///     real workspace vector of size equal to the number of
    ///     variables in the reduced problem.
    ipc_ check_dual_feasibility;

    /// \brief
    /// The relative pivot tolerance above which pivoting is
    /// considered as numerically stable in transforming the
    /// coefficient matrix A.  A zero value corresponds to a
    /// totally unsafeguarded pivoting strategy (potentially
    /// unstable).
    rpc_ pivot_tol;

    /// \brief
    /// The minimum relative improvement in the bounds on x, y
    /// and z for a tighter bound on these quantities to be
    /// accepted in the course of the analysis.  More formally,
    /// if lower is the current value of the lower bound on one
    /// of the x, y or z, and if new_lower is a tentative tighter
    /// lower bound on the same quantity, it is only accepted
    /// if
    ///
    ///  new_lower >= lower + tol * MAX( 1, ABS( lower ) ),
    ///
    /// where
    ///
    /// tol = control.min_rel_improve.
    ///
    /// Similarly, a tentative tighter upper bound new_upper
    /// only replaces the current upper bound upper if
    ///
    ///  new_upper <= upper - tol * MAX( 1, ABS( upper ) ).
    ///
    /// Note that this parameter must exceed the machine
    /// precision significantly.
    rpc_ min_rel_improve;

    /// \brief
    /// The maximum growth factor (in absolute value) that is
    /// accepted between the maximum data item in the original
    /// problem  and any data item in the reduced problem.
    /// If a transformation results in this bound being exceeded,
    /// the transformation is skipped.
    rpc_ max_growth_factor;
};

/**
 * inform derived type as a C struct
 */
struct presolve_inform_type {

    /// \brief
    /// The presolve exit condition.  It can take the following
    /// values (symbol in parentheses is the related Fortran code):
    /// \li (OK)
    ///    successful exit;
    /// \li 1 (MAX_NBR_TRANSF)
    ///   the maximum number of problem transformation has been reached
    ///   NOTE:
    ///   this exit is not really an error, since the problem
    ///   can  nevertheless be permuted and  solved.  It merely
    ///   signals that further problem reduction could possibly
    ///   be obtained with a larger value of the parameter
    ///   \p control.max_nbr_transforms
    /// \li -21 (PRIMAL_INFEASIBLE)
    ///   the problem is primal infeasible;
    /// \li -22 (DUAL_INFEASIBLE)
    ///   the problem is dual infeasible;
    /// \li -23 (WRONG_G_DIMENSION)
    ///   the dimension of the gradient is incompatible with
    ///   the problem dimension;
    /// \li -24 (WRONG_HVAL_DIMENSION)
    ///   the dimension of the vector containing the entries of
    ///   the Hessian is erroneously specified;
    /// \li -25 (WRONG_HPTR_DIMENSION)
    ///   the dimension of the vector containing the addresses
    ///   of the first entry of each Hessian row is erroneously specified;
    /// \li -26 (WRONG_HCOL_DIMENSION)
    ///   the dimension of the vector containing the column
    ///   indices of the nonzero Hessian entries is erroneously specified;
    /// \li -27 (WRONG_HROW_DIMENSION)
    ///   the dimension of the vector containing the row indices
    ///   of the nonzero Hessian entries is erroneously specified;
    /// \li -28 (WRONG_AVAL_DIMENSION)
    ///   the dimension of the vector containing the entries of
    ///   the Jacobian is erroneously specified;
    /// \li -29 (WRONG_APTR_DIMENSION)
    ///   the dimension of the vector containing the addresses
    ///   of the first entry of each Jacobian row is erroneously specified;
    /// \li -30 (WRONG_ACOL_DIMENSION)
    ///   the dimension of the vector containing the column
    ///   indices of the nonzero Jacobian entries is erroneously specified;
    /// \li -31 (WRONG_AROW_DIMENSION)
    ///   the dimension of the vector containing the row indices
    ///   of the nonzero Jacobian entries is erroneously specified;
    /// \li -32 (WRONG_X_DIMENSION)
    ///   the dimension of the vector of variables is
    ///   incompatible with the problem dimension;
    /// \li -33 (WRONG_XL_DIMENSION)
    ///   the dimension of the vector of lower bounds on the
    ///   variables is incompatible with the problem dimension;
    /// \li -34 (WRONG_XU_DIMENSION)
    ///   the dimension of the vector of upper bounds on the
    ///   variables is incompatible with the problem dimension;
    /// \li -35 (WRONG_Z_DIMENSION)
    ///   the dimension of the vector of dual variables is
    ///   incompatible with the problem dimension;
    /// \li -36 (WRONG_ZL_DIMENSION)
    ///   the dimension of the vector of lower bounds on the dual
    ///   variables is incompatible with the problem dimension;
    /// \li -37 (WRONG_ZU_DIMENSION)
    ///   the dimension of the vector of upper bounds on the
    ///   dual variables is incompatible with the problem dimension;
    ipc_ status;

    /// \brief
    /// continuation of status (name in previous column should be status,
    /// doxygen issue):
    /// \li -38 (WRONG_C_DIMENSION)
    ///   the dimension of the vector of constraints values is
    ///   incompatible with the problem dimension;
    /// \li -39 (WRONG_CL_DIMENSION)
    ///   the dimension of the vector of lower bounds on the
    ///   constraints is incompatible with the problem dimension;
    /// \li -40 (WRONG_CU_DIMENSION)
    ///   the dimension of the vector of upper bounds on the
    ///   constraints is incompatible with the problem dimension;
    /// \li -41 (WRONG_Y_DIMENSION)
    ///   the dimension of the vector of multipliers values is
    ///   incompatible with the problem dimension;
    /// \li -42 (WRONG_YL_DIMENSION)
    ///   the dimension of the vector of lower bounds on the
    ///   multipliers is incompatible with the problem dimension;
    /// \li -43 (WRONG_YU_DIMENSION)
    ///   the dimension of the vector of upper bounds on the
    ///   multipliers is incompatible with the problem dimension;
    /// \li -44 (STRUCTURE_NOT_SET)
    ///   the problem structure has not been set or has been
    ///   cleaned up before an attempt to analyze;
    /// \li -45 (PROBLEM_NOT_ANALYZED)
    ///   the problem has not been analyzed before an attempt to permute it;
    /// \li -46 (PROBLEM_NOT_PERMUTED)
    ///   the problem has not been permuted or fully reduced before an attempt
    ///   to restore it
    /// \li -47 (H_MISSPECIFIED)
    ///   the column indices of a row of the sparse Hessian are
    ///   not in increasing order, in that they specify an entry
    ///   above the diagonal;
    /// \li -48 (CORRUPTED_SAVE_FILE)
    ///   one of the files containing saved problem
    ///   transformations has been corrupted between writing and reading;
    /// \li -49 (WRONG_XS_DIMENSION)
    ///   the dimension of the vector of variables' status
    ///   is incompatible with the problem dimension;
    /// \li -50 (WRONG_CS_DIMENSION)
    ///   the dimension of the vector of constraints' status
    ///   is incompatible with the problem dimension;
    /// \li -52 (WRONG_N)
    ///   the problem does not contain any (active) variable;
    /// \li -53 (WRONG_M)
    ///   the problem contains a negative number of constraints;
    /// \li -54 (SORT_TOO_LONG)
    ///   the vectors are too long for the sorting routine;
    /// \li -55 (X_OUT_OF_BOUNDS)
    ///   the value of a variable that is obtained by
    ///   substitution from a constraint is incoherent with the
    ///   variable's bounds.  This may be due to a relatively
    ///   loose accuracy on the linear constraints. Try to
    ///   increase control.c_accuracy.
    /// \li -56 (X_NOT_FEASIBLE)
    ///   the value of a constraint that is obtained by
    ///   recomputing its value on input of \p presolve_restore_solution
    ///   from the current x is incompatible with its declared value
    ///   or its bounds. This may caused the restored problem
    ///   to be infeasible.
    /// \li -57 (Z_NOT_FEASIBLE)
    ///   the value of a dual variable that is obtained by
    ///   recomputing its value on input to \p presolve_restore_solution
    ///   (assuming dual feasibility) from the current values of
    ///   \f$(x, y, z)\f$ is incompatible with its declared value.
    ///   This may caused the restored problem to be infeasible
    ///   or suboptimal.
    ipc_ status_continue;

    /// \brief
    /// continuation of status (name in previous column should be status,
    /// doxygen issue):
    /// \li -58 (Z_CANNOT_BE_ZEROED)
    ///   a dual variable whose value is nonzero because the
    ///   corresponding primal is at an artificial bound cannot
    ///   be zeroed while maintaining dual feasibility
    ///   (on restoration). This can happen when \f$( x, y, z)\f$ on
    ///   input of RESTORE are not (sufficiently) optimal.
    /// \li -1 (MEMORY_FULL)
    ///   memory allocation failed
    /// \li -2 (FILE_NOT_OPENED)
    ///   a file intended for saving problem transformations
    ///   could not be opened;
    /// \li -3 (COULD_NOT_WRITE)
    ///   an IO error occurred while saving transformations on
    ///   the relevant disk file;
    /// \li -4 (TOO_FEW_BITS_PER_BYTE)
    ///   an integer contains less than NBRH + 1 bits.
    /// \li -60 (UNRECOGNIZED_KEYWORD)
    ///   a keyword was not recognized in the analysis of the
    ///   specification file
    /// \li -61 (UNRECOGNIZED_VALUE)
    ///   a value was not recognized in the analysis of the specification file
    /// \li -63 (G_NOT_ALLOCATED)
    ///   the vector G has not been allocated although it has general values
    /// \li -64 (C_NOT_ALLOCATED)
    ///   the vector C has not been allocated although m > 0
    /// \li -65 (AVAL_NOT_ALLOCATED)
    ///   the vector A.val has not been allocated although m > 0
    /// \li -66 (APTR_NOT_ALLOCATED)
    ///   the vector A.ptr has not been allocated although
    ///   m > 0 and A is stored in row-wise sparse format
    /// \li -67 (ACOL_NOT_ALLOCATED)
    ///   the vector A.col has not been allocated although
    ///   m > 0 and A is stored in row-wise sparse format
    ///   or sparse coordinate format
    /// \li -68 (AROW_NOT_ALLOCATED)
    ///   the vector A.row has not been allocated although
    ///   m > 0 and A is stored in sparse coordinate format
    /// \li -69 (HVAL_NOT_ALLOCATED)
    ///   the vector H.val has not been allocated although
    ///   H.ne > 0
    /// \li -70 (HPTR_NOT_ALLOCATED)
    ///   the vector H.ptr has not been allocated although
    ///   H.ne > 0 and H is stored in row-wise sparse format
    /// \li -71 (HCOL_NOT_ALLOCATED)
    /// the vector H.col has not been allocated although
    ///   H.ne > 0 and H is stored in row-wise sparse format
    ///   or sparse coordinate format
    /// \li -72 (HROW_NOT_ALLOCATED)
    ///   the vector H.row has not been allocated although
    ///   H.ne > 0 and A is stored in sparse coordinate
    ///   format
    /// \li -73 (WRONG_ANE)
    ///   incompatible value of A_ne
    /// \li -74 (WRONG_HNE)
    ///   incompatible value of H_ne
    ipc_ status_continued;

    /// \brief
    /// The final number of problem transformations, as reported
    /// to the user at exit.
    ipc_ nbr_transforms;

    /// \brief
    /// A few lines containing a description of the exit condition
    /// on exit of PRESOLVE, typically including more information
    /// than indicated in the description of control.status above.
    /// It is printed out on device errout at the end of execution
    /// if control.print_level >= 1.
    char message[3][81];
};

// *-*-*-*-*-*-*-    P R E S O L V E  _ I N I T I A L I Z E    -*-*-*-*-*-*-

void presolve_initialize( void **data,
                     struct presolve_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see presolve_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-    P R E S O L V E  _ R E A D _ S P E C F I L E   -*-*-*-*-*-

void presolve_read_specfile( struct presolve_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information
              (see presolve_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-   P R E S O L V E _ I M P O R T _ P R O B L E M  -*-*-*-*-*-*-

void presolve_import_problem( struct presolve_control_type *control,
                              void **data,
                              ipc_ *status,
                              ipc_ n,
                              ipc_ m,
                              const char H_type[],
                              ipc_ H_ne,
                              const ipc_ H_row[],
                              const ipc_ H_col[],
                              const ipc_ H_ptr[],
                              const rpc_ H_val[],
                              const rpc_ g[],
                              const rpc_ f,
                              const char A_type[],
                              ipc_ A_ne,
                              const ipc_ A_row[],
                              const ipc_ A_col[],
                              const ipc_ A_ptr[],
                              const rpc_ A_val[],
                              const rpc_ c_l[],
                              const rpc_ c_u[],
                              const rpc_ x_l[],
                              const rpc_ x_u[],
                              ipc_ *n_out,
                              ipc_ *m_out,
                              ipc_ *H_ne_out,
                              ipc_ *A_ne_out );

/*!<
 Import the initial data, and apply the presolve algorithm to report
 crucial characteristics of the transformed variant

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see presolve_control_type)

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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows' or
       'diagonal' has been violated.
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

  @param[in] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

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

 @param[out] n_out is a scalar variable of type ipc_, that holds the number of
    variables in the transformed problem.

 @param[out] m_out is a scalar variable of type ipc_, that holds the number of
    general linear constraints in the transformed problem.

 @param[out]  H_ne_out is a scalar variable of type ipc_, that holds the number
   of entries in the lower triangular part of \f$H\f$ in the transformed
   problem.

 @param[out]  A_ne_out is a scalar variable of type ipc_, that holds the number
   of entries in \f$A\f$ in the transformed problem.

*/

// *-*-*-*-*-   P R E S O L V E _ T R A N S F O R M _ P R O B L E M  -*-*-*-*-

void presolve_transform_problem( void **data,
                                 ipc_ *status,
                                 ipc_ n,
                                 ipc_ m,
                                 ipc_ H_ne,
                                 ipc_ H_col[],
                                 ipc_ H_ptr[],
                                 rpc_ H_val[],
                                 rpc_ g[],
                                 rpc_ *f,
                                 ipc_ A_ne,
                                 ipc_ A_col[],
                                 ipc_ A_ptr[],
                                 rpc_ A_val[],
                                 rpc_ c_l[],
                                 rpc_ c_u[],
                                 rpc_ x_l[],
                                 rpc_ x_u[],
                                 rpc_ y_l[],
                                 rpc_ y_u[],
                                 rpc_ z_l[],
                                 rpc_ z_u[] );

/*!<
 Apply the presolve algorithm to simplify the input problem, and
 output the transformed variant

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
  \li -3. The input values n, m, A_ne or H_ne do not agree with those
      output as necessary from presolve_import_problem.

 @param[out] n is a scalar variable of type ipc_, that holds the number of
    variables in the transformed problem. This must match the value
    n_out from the last call to presolve_import_problem.

 @param[out] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints. This must match the value
    m_out from the last call to presolve_import_problem.

 @param[out] H_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of the transformed \f$H\f$.
   This must match the value H_ne_out from the last call to
   presolve_import_problem.

 @param[out] H_col is a one-dimensional array of size H_ne and type ipc_,
   that holds the column indices of the lower triangular part of the
   transformed \f$H\f$ in the sparse row-wise storage scheme.

 @param[out] H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of the transformed \f$H\f$ in
   the sparse row-wise storage scheme.

  @param[out] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    the transformed Hessian matrix \f$H\f$ in
    the sparse row-wise storage scheme.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    holds the the transformed linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[out] f is a scalar of type rpc_, that
    holds the transformed constant term \f$f\f$ of the objective function.

 @param[out] A_ne is a scalar variable of type ipc_, that holds the number of
   entries in the transformed \f$A\f$.  This must match the value A_ne_out
   from the last call to presolve_import_problem.

 @param[out]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of the transformed \f$A\f$ in the
   sparse row-wise storage scheme.

 @param[out]  A_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each row of the transformed \f$A\f$,
   as well as the total number of entries, in the sparse row-wise
   storage scheme.

 @param[out] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the transformed constraint
    Jacobian matrix \f$A\f$ in the sparse row-wise storage scheme.

 @param[out] c_l is a one-dimensional array of size m and type rpc_, that
    holds the transformed lower bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_l, i = 0, ... ,  m-1, contains  \f$c^l_i\f$.

 @param[out] c_u is a one-dimensional array of size m and type rpc_, that
    holds the transformed upper bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_u, i = 0, ... ,  m-1, contains  \f$c^u_i\f$.

 @param[out] x_l is a one-dimensional array of size n and type rpc_, that
    holds the transformed lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[out] x_u is a one-dimensional array of size n and type rpc_, that
    holds the transformed upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[out] y_l is a one-dimensional array of size m and type rpc_, that
    holds the implied lower bounds \f$y^l\f$ on the transformed Lagrange
    multipliers \f$y\f$.
    The i-th component of y_l, i = 0, ... ,  m-1, contains  \f$y^l_i\f$.

 @param[out] y_u is a one-dimensional array of size m and type rpc_, that
    holds the implied upper bounds \f$y^u\f$ on the transformed Lagrange
    multipliers \f$y\f$.
    The i-th component of y_u, i = 0, ... ,  m-1, contains  \f$y^u_i\f$.

 @param[out] z_l is a one-dimensional array of size m and type rpc_, that
    holds the implied lower bounds \f$y^l\f$ on the transformed dual variables
    \f$z\f$.
    The j-th component of z_l, j = 0, ... ,  n-1, contains  \f$z^l_i\f$.

 @param[out] z_u is a one-dimensional array of size m and type rpc_, that
    holds the implied upper bounds \f$y^u\f$ on the transformed dual variables
    \f$z\f$.
    The j-th component of z_u, j = 0, ... ,  n-1, contains  \f$z^u_i\f$.

*/

// *-*-*-*-*-   P R E S O L V E _ R E S T O R E + S O L U T I O N  -*-*-*-*-*-

void presolve_restore_solution( void **data,
                                ipc_ *status,
                                ipc_ n_in,
                                ipc_ m_in,
                                const rpc_ x_in[],
                                const rpc_ c_in[],
                                const rpc_ y_in[],
                                const rpc_ z_in[],
                                ipc_ n,
                                ipc_ m,
                                rpc_ x[],
                                rpc_ c[],
                                rpc_ y[],
                                rpc_ z[] );

/*!<
 Given the solution (x_in,c_in,y_in,z_in) to the transformed problem,
 restore to recover the solution (x,c,y,z) to the original

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
  \li -3. The input values n, m, n_in and m_in do not agree with those
      input to and output as necessary from presolve_import_problem.

 @param[out] n_in is a scalar variable of type ipc_, that holds the number of
    variables in the transformed problem. This must match the value
    n_out from the last call to presolve_import_problem.

 @param[out] m_in is a scalar variable of type ipc_, that holds the number of
    general linear constraints. This must match the value
    m_out from the last call to presolve_import_problem.

 @param[in] x_in is a one-dimensional array of size n_in and type rpc_, that
    holds the transformed values \f$x\f$ of the optimization variables.
    The j-th component of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in] c_in is a one-dimensional array of size m and type rpc_, that
    holds the transformed residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.

 @param[in] y_in is a one-dimensional array of size n_in and type rpc_, that
    holds the values \f$y\f$ of the transformed Lagrange multipliers for
    the general linear constraints. The j-th component
    of y, j = 0, ... , n-1, contains \f$y_j\f$.

 @param[in] z_in is a one-dimensional array of size n_in and type rpc_, that
    holds the values \f$z\f$ of the transformed dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables in the transformed problem. This must match the value
    n as input to presolve_import_problem.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints. This must match the value
    m as input to presolve_import_problem.

 @param[out] x is a one-dimensional array of size n and type rpc_, that
    holds the transformed values \f$x\f$ of the optimization variables.
    The j-th component of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[out] c is a one-dimensional array of size m and type rpc_, that
    holds the transformed residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.

 @param[out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the transformed Lagrange multipliers for
    the general linear constraints. The j-th component
    of y, j = 0, ... , n-1, contains \f$y_j\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the transformed dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.
*/


// *-*-*-*-*-*-*-    P R E S O L V E  _ I N F O R M A T I O N   -*-*-*-*-*-*-

void presolve_information( void **data,
                      struct presolve_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see presolve_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-    P R E S O L V E  _ T E R M I N A T E   -*-*-*-*-*-*-*-

void presolve_terminate( void **data,
                    struct presolve_control_type *control,
                    struct presolve_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see presolve_control_type)

  @param[out] inform   is a struct containing output information
              (see presolve_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example presolvet.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example presolvetf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
