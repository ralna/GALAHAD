/** \file galahad_expo.h */

/*
 * THIS VERSION: GALAHAD 5.3 - 2025-07-25 AT 08:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_EXPO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.3. July 23rd 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package expo

  \section expo_intro Introduction

  \subsection expo_purpose Purpose

  This package uses a <b>regularization method to find a (local)
  minimizer of a objective \f$f(x)\f$, where the
  variables \f$x\f$ are required to satisfy the general constraints
  \f[c_i^l  \leq  c_i(x)  \leq c_i^u, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
   c_i^l \[<=] c_i(x) \[<=] c_i^u, i = 1, ... , m,
  \n
\endmanonly
  and the simple bounds
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
  where the vectors \f$c^l\f$, \f$c^u\f$, \f$x^l\f$ and
  \f$x^u\f$ are given;
  any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in constraint Jacobian
  matrix \f$J(x)\f$ of vectors \f$\nabla c_i(x)\f$ or the Hessian of the
  Lagrangian \f$H(x,y) = \nabla^2 f(x) - \sum_{i=1}^m y_i  \nabla^2 c_i(x)\f$.
  The method offers the choice of direct and iterative solution of the key
  regularization subproblems, and is most suitable for large problems.
  First derivatives (the gradient, Jacobian) of the <i>objective and
  constraint function</i> \f$f(x)\f$ and  \f$c(x)\f$  are required,
  and if second derivatives of the \f$c_i(x)\f$ can be calculated,
  they may be exploited---if suitable products of the first or second
  derivatives with a vector may be found but not the
  derivatives themselves, that can also be used to advantage (forthcoming).

  \subsection expo_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection expo_date Originally released

  July 2025 (beta), C interface July 2025 (beta)

  \subsection expo_terminology Terminology

  To be done ...
  The \e gradient \f$\nabla_x f(x)\f$ of a function \f$f(x)\f$ is the vector
  whose \f$i\f$-th component is \f$\partial f(x)/\partial x_i\f$.
  The \e Hessian \f$\nabla_{xx} f(x)\f$ of \f$f(x)\f$ is the symmetric matrix
  whose \f$i,j\f$-th entry is \f$\partial^2 f(x)/\partial x_i \partial x_j\f$.
  The Hessian is \e sparse if a significant and useful proportion of the
  entries are universally zero.

  The algorithm used by the package is iterative. From the current best estimate
  of the minimizer \f$x_k\f$, a trial improved point \f$x_k + s_k\f$ is sought.
  The correction \f$s_k\f$ is chosen to improve a model \f$m_k(s)\f$ of
  %the stabilised objective function \f$f_{\rho,p}(x_k+s)\f$ built around
  the objective function \f$f(x_k+s)\f$ built around
  \f$x_k\f$. The model is the sum of two basic components,
  a suitable approximation \f$t_k(s)\f$ of \f$f(x_k+s)\f$,
  %another approximation of \f$(\rho/r) \|x_k+s\|_r^r\f$
  (if \f$\rho > 0\f$),
  and a regularization term \f$(\sigma_k/p) \|s\|_{S_k}^p\f$
  involving a weight \f$\sigma_k\f$, power \f$p\f$ and
  a norm \f$\|s\|_{S_k} := \sqrt{s^T S_k s}\f$ for a given positive
  definite scaling matrix \f$S_k\f$ that is included to prevent large
  corrections. The weight  \f$\sigma_k\f$ is adjusted as the algorithm
  progresses to  ensure convergence.

  The model \f$t_k(s)\f$ is a truncated Taylor-series approximation, and this
  relies on being able to compute or estimate derivatives of \f$c(x)\f$.
  Various models are provided, and each has different derivative requirements.
  We denote the \f$m\f$ by \f$n\f$ <i>residual Jacobian</i>
  \f$J(x) \equiv \nabla_x c(x)\f$ as the matrix  whose \f$i,j\f$-th component
  \f[J(x)_{i,j} := \partial c_i(x) / \partial x_j \;\;
  \mbox{for $i=1,\ldots,m$ and $j=1,\ldots,n$.}\f]
\manonly
  \n
  J(x)_{i,j} := partial c_i(x) / \partial x_j
  \n
\endmanonly
\manonly for i=1,...,m and j=1,...,n.\endmanonly
  For a given \f$m\f$-vector \f$y\f$, the
  <i>weighted-residual Hessian</i> is the sum
  \f[H(x,y) := \sum_{\ell=1}^m y_\ell H_\ell(x),
  \;\; \mbox{where}\;\;
  H_\ell(x)_{i,j} := \partial^2 c_\ell(x) / \partial x_i \partial x_j
  \;\; \mbox{for $i,j=1,\ldots,n$}\f]
\manonly
  \n
  H(x,y) := sum_{\ell=1}^m y_\ell H_\ell(x), where
  \n
  H_l(x)_{i,j} := partial^2 c_l(x) / partial x_i partial x_j
  \n
for i,j=1,...,n\endmanonly
  is the Hessian of \f$c_\ell(x)\f$.
  Finally, for a given vector \f$v\f$, we define
  the <i>residual-Hessians-vector product matrix</i>
  \f[P(x,v) := (H_1(x) v, \ldots, H_m(x) v).\f]
\manonly
  \n
  P(x,v) := (H_1(x) v, ..., H_m(x) v).
  \n
\endmanonly
  The models \f$t_k(s)\f$ provided are,
  -# the first-order Taylor approximation \f$f(x_k) + g(x_k)^T s\f$,
  where
  \f$g(x) = J^T(x) W c(x)\f$,
  -# a barely second-order approximation
  \f$f(x_k) + g(x_k)^T s + \frac{1}{2} s^T W s\f$,
  -# the Gauss-Newton approximation
  \f$\frac{1}{2} \| c(x_k) + J(x_k) s\|^2_W\f$,
  -# the Newton (second-order Taylor) approximation
    \f$f(x_k) + g(x_k)^T s + \frac{1}{2} s^T
    [ J^T(x_k) W J(x_k) + H(x_k,W c(x_k))] s\f$, and
  -# the tensor Gauss-Newton approximation
  \f$\frac{1}{2} \| c(x_k) + J(x_k) s +
   \frac{1}{2} s^T \cdot P(x_k,s) \|^2_W\f$,
  where the \f$i\f$-th component of \f$s^T \cdot P(x_k,s)\f$ is
  shorthand for the scalar \f$s^T H_i(x_k) s\f$,
  where \f$W\f$ is the diagonal matrix of weights
  \f$w_i\f$, \f$i = 1, \ldots m\f$.

  Access to a particular model requires that the user is either able to
  provide the derivatives needed (``<i>matrix available</i>'')
  or that the products
  of these derivatives (and their transposes) with specified vectors are
  possible (``<i>matrix free</i>'').

  \subsection expo_method Method

  To be done ...
  An adaptive regularization method is used.
  In this, an improvement to a current
  estimate of the required minimizer, \f$x_k\f$ is sought by computing a
  step \f$s_k\f$. The step is chosen to approximately minimize a model
  \f$t_k(s)\f$ of \f$f_{\rho,r}(x_k+s)\f$
  that includes a weighted regularization term
  \f$(\sigma_k/p) \|s\|_{S_k}^p\f$
  for some specified positive weight \f$\sigma_k\f$. The quality of the
  resulting step \f$s_k\f$ is assessed by computing the "ratio"
  %\f$(f_{\rho,p}(x_k) - f_{\rho,p}(x_k+s_k))/(t_k(0)-t_k(s_k))\f$.
  \f$(f(x_k) - f(x_k + s_k))/(t_k(0) - t_k(s_k))\f$.
  The step is deemed to have succeeded if the ratio exceeds a given
  \f$\eta_s > 0\f$,
  and in this case \f$x_{k+1} = x_k + s_k\f$. Otherwise
  \f$x_{k+1} = x_k\f$, and the weight is increased by powers of a given
  increase factor up to a given limit. If the ratio is larger than
  \f$\eta_v \geq \eta_d\f$, the weight will be decreased by powers of a given
  decrease factor again up to a given limit. The method will terminate
  as soon as \f$f(x_k)\f$ or
  \f$\|\nabla_x f(x_k)\|\f$ is smaller than a specified value.

  A choice of linear, quadratic or quartic models \f$t_k(s)\f$ is available
  (see the \ref expo_terminology section), and normally a two-norm
  regularization will  be used, but this may change if preconditioning
  is employed.

  If linear or quadratic models are employed, an appropriate,
  approximate model minimizer is found using either a direct approach
  involving factorization of a shift of the model Hessian \f$B_k\f$ or an
  iterative (conjugate-gradient/Lanczos) approach based on approximations
  to the required solution from a so-called Krlov subspace. The direct
  approach is based on the knowledge that the required solution
  satisfies the linear system of equations \f$(B_k + \lambda_k I) s_k
  = - \nabla_x f(x_k)\f$ involving a scalar Lagrange multiplier \f$\lambda_k\f$.
  This multiplier is found by uni-variate root finding, using a safeguarded
  Newton-like process, by the GALAHAD packages TRU. The iterative approach
  uses the GALAHAD packag GLRT, and is best accelerated by preconditioning
  with good approximations to the Hessian of the model using GALAHAD's PSLS. The
  iterative approach has the advantage that only Hessian matrix-vector products
  are required, and thus the Hessian \f$B_k\f$ is not required explicitly.
  However when factorizations of the Hessian are possible, the direct approach
  is often more efficient.

  When a quartic model is used, the model is itself of least-squares form,
  and the package calls itself recursively to approximately minimize its
  model. The quartic model often gives a better approximation, but at the
  cost of more involved derivative requirements.

  \subsection expo_references Reference

  The generic adaptive cubic regularization method is described in detail in

  C. Cartis,  N. I. M. Gould and Ph. L. Toint,
  ``Adaptive cubic regularisation methods for unconstrained optimization.
  Part I: motivation, convergence and numerical results'',
  Mathematical Programming 127(2) (2011) 245-295,

  and uses ``tricks'' as suggested in

  N. I. M. Gould, M. Porcelli and Ph. L. Toint,
  ``Updating the regularization parameter in the adaptive cubic regularization
  algorithm''.
  Computational Optimization and Applications 53(1) (2012) 1-22.

  \subsection expo_call_order Call order

  To solve a given problem, functions from the expo package must be called
  in the following order:

  - \link expo_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link expo_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link expo_import \endlink - set up problem data structures and fixed
      values
  - \link expo_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link expo_solve_with_mat \endlink - solve using function calls to
       evaluate function, gradient and Hessian values
     - \link expo_solve_without_mat \endlink - solve using function calls to
       evaluate function and gradient values and Hessian-vector products
     - \link expo_solve_reverse_with_mat \endlink - solve returning to the
       calling program to obtain function, gradient and Hessian values, or
     - \link expo_solve_reverse_without_mat \endlink - solve returning to the
       calling prorgram to obtain function and gradient values and
       Hessian-vector products
  - \link expo_information \endlink (optional) - recover information about
    the solution and solution process
  - \link expo_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$m\f$ by \f$n\f$ Jacobian matrix
  \f$J \equiv \nabla_x c(x)\f$ and the residual-Hessians-vector
  product matrix $P(x,v)$ may be presented
  and stored in a variety of convenient input formats. Let
  \f$A\f$ be \f$J\f$ or \f$P\f$ as appropriate.

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

  \subsubsection unsymmetric_matrix_dense_cols Dense by columns storage format

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
  in column j+1. For the j-th column of \f$A\f$ the j-th component of the
  integer array A_ptr holds the position of the first entry in this column,
  while A_ptr(n) holds the total number of entries.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values \f$A_{ij}\f$
  of the  nonzero entries in the j-th columnsare stored in components
  l = A_ptr(j), \f$\ldots\f$, A_ptr(j+1)-1, \f$0 \leq j \leq n-1\f$,
  of the integer array A_row, and real array A_val, respectively.
  As before, for sparse matrices, this scheme almost always requires less
  storage than the co-ordinate format.

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  Likewise, the symmetric \f$n\f$ by \f$n\f$ weighted-residual
  Hessian matrix \f$H = H(x,y)\f$ may be presented
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
#ifndef GALAHAD_EXPO_H
#define GALAHAD_EXPO_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_bsc.h"
#include "galahad_tru.h"
#include "galahad_ssls.h"

/**
 * control derived type as a C struct
 */
struct expo_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief the level of output required.
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
    /// the number of iterations between printing
    ipc_ print_gap;

    /// \brief
    /// the maximum number of iterations permitted
    ipc_ max_it;

    /// \brief
    /// the maximum number of function evaluations permitted
    ipc_ max_eval;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    ipc_ alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// update the Lagrange multipliers/dual variables from iteration
    /// update_multipliers_itmin (<0 means never) and once the primal
    /// infeasibility is below update_multipliers_tol
    ipc_ update_multipliers_itmin;
    /// see update_multipliers_itmin
    rpc_ update_multipliers_tol;

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
    /// the smallest the norm of the step can be before termination
    rpc_ stop_s;

    /// \brief
    /// initial value for the penalty parameter (<=0 means set automatically)
    rpc_ initial_mu;

    /// \brief
    /// the amount by which the penalty parameter is decreased
    rpc_ mu_reduce;

    /// \brief
    /// if the objective function value is smaller than obj_unbounded, it will
    /// be flagged as unbounded from below.
    rpc_ obj_unbounded;

    /// \brief
    ///  try an advanced start at the end of every iteration when the KKT
    ///  residuals are smaller than .try_advanced_start (-ve means never)
    rpc_ try_advanced_start;

    /// \brief
    /// try an advanced SQP start at the end of every iteration when the KKT
    /// residuals are smaller than .try_sqp_start (-ve means never)
    rpc_ try_sqp_start;

    /// \brief
    /// stop the advanced start search once the residuals small than
    /// .stop_advanced_start
    rpc_ stop_advanced_start;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// is the Hessian matrix of second derivatives available or is access only
    /// via matrix-vector products (coming soon)
    bool hessian_available;

    /// \brief
    /// use a direct (factorization) or (preconditioned) iterative method
    /// (coming soon) to find the search direction
    bool subproblem_direct;

    /// \brief
    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for BSC
    struct bsc_control_type bsc_control;

    /// \brief
    /// control parameters for TRU
    struct tru_control_type tru_control;

    /// \brief
    /// control parameters for SSLS
    struct ssls_control_type ssls_control;
};

/**
 * time derived type as a C struct
 */
struct expo_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    real_sp_ preprocess;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to
    /// factorization
    real_sp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_sp_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    real_sp_ solve;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    rpc_ clock_preprocess;

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
struct expo_inform_type {

    /// \brief
    /// return status. See EXPO_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
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
    /// the total number of evaluations of the objective f(x)  and constraint
    /// c(x) functions
    ipc_ fc_eval;

    /// \brief
    /// the total number of evaluations of the gradient g(x) and Jacobian J(x)
    ipc_ gj_eval;

    /// \brief
    /// the total number of evaluations of the Hessian H(x,y) of the Lagrangian
    ipc_ hl_eval;

    /// \brief
    /// the value of the objective function \f$f(x)\f$
    /// at the best estimate the solution, x, determined by EXPO_solve
    rpc_ obj;

    /// \brief
    /// the norm of the primal infeasibility at the best estimate of
    /// the solution x, determined by EXPO_solve
    rpc_ primal_infeasibility;

    /// \brief
    /// the norm of the dual infeasibility at the best estimate of
    /// the solution x, determined by EXPO_solve
    rpc_ dual_infeasibility;

    /// \brief
    /// the norm of the complementary slackness at the best estimate of
    /// the solution x, determined by EXPO_solve
    rpc_ complementary_slackness;

    /// \brief
    /// timings (see above)
    struct expo_time_type time;

    /// \brief
    /// inform parameters for BSC
    struct bsc_inform_type bsc_inform;

    /// \brief
    /// inform parameters for TRU
    struct tru_inform_type tru_inform;

    /// \brief
    /// inform parameters for SSLS
    struct ssls_inform_type ssls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    E X P O  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void expo_initialize( void **data,
                     struct expo_control_type *control,
                     struct expo_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information
              (see expo_control_type)
  @param[out] inform   is a struct containing output information
              (see expo_inform_type)
*/

// *-*-*-*-*-*-*-*-*-    E X P O  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void expo_read_specfile( struct expo_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNEXPO.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/expo.pdf for a list of keywords that may be set.

  @param[in,out]  control  is a struct containing control information
              (see expo_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    E X P O  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void expo_import( struct expo_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char J_type[],
                 ipc_ J_ne,
                 const ipc_ J_row[],
                 const ipc_ J_col[],
                 const ipc_ J_ptr[],
                 const char H_type[],
                 ipc_ H_ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see expo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0, m > 0 or requirement that J/H/P_type
       contains its relevant string 'dense', 'dense_by_columns',
       'coordinate', 'sparse_by_rows', 'sparse_by_columns',
       'diagonal' or 'absent' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    constraints.

 @param[in]  J_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the Jacobian, \f$J\f$. It should be one of 'coordinate',
  'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian
  is via matrix-vector products; lower or upper case variants are allowed.

 @param[in]  J_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$J\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes.

 @param[in]  J_row is a one-dimensional array of size J_ne and type ipc_, that
   holds the row indices of \f$J\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes,
   and in this case can be NULL.

 @param[in]  J_col is a one-dimensional array of size J_ne and type ipc_,
   that holds the column indices of \f$J\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  J_ptr is a one-dimensional array of size m+1 and type ipc_,
   that holds the starting position of each row of \f$J\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the Hessian, \f$H\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to
   \f$H\f$ is via matrix-vector products; lower or upper case variants
    are allowed.

 @param[in]  H_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$H\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other three schemes.

 @param[in]  H_row is a one-dimensional array of size H_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$H\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  H_col is a one-dimensional array of size H_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$H\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense or diagonal storage schemes are used,
   and in this case can be NULL.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
 */

//  *-*-*-*-*-*-*-*-*-   E X P O _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void expo_reset_control( struct expo_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see expo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
 */

//  *-*-*-*-*-*-   E X P O _ S O L V E _ H E S S I A N _D I R E C T   -*-*-*-*-

void expo_solve_hessian_direct( void **data,
                                void *userdata,
                                ipc_ *status,
                                ipc_ n,
                                ipc_ m,
                                ipc_ J_ne,
                                ipc_ H_ne,
                                const rpc_ c_l[],
                                const rpc_ c_u[],
                                const rpc_ x_l[],
                                const rpc_ x_u[],
                                rpc_ x[],
                                rpc_ y[],
                                rpc_ z[],
                                rpc_ c[],
                                rpc_ gl[],
                                ipc_ (*eval_fc)(
                                  ipc_, ipc_, const rpc_[],
                                  rpc_, rpc_[], const void * ),
                                ipc_ (*eval_gj)(
                                  ipc_, ipc_, ipc_, const rpc_[],
                                  rpc_[], rpc_[], const void * ),
                                ipc_ (*eval_hl)(
                                  ipc_, ipc_, ipc_, const rpc_[], const rpc_[],
                                  rpc_[], const void * ) );

/*!<
 Find a local minimizer of the constrained optimization problem using the
 exponential penalty method.

 This call is for the case where the Hessian of the Lagrangian function is
 available specifically, and all function/derivative information is
 available by (direct) function calls.

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
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    constraints.

 @param[in]  J_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$J\f$.

 @param[in]  H_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$H\f$.

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

 @param[out] y is a one-dimensional array of size m and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers. The i-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables. The j-th component
    of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] c is a one-dimensional array of size m and type rpc_, that
    holds the values of the constraints \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.

 @param[out] gl is a one-dimensional array of size n and type rpc_, that
    holds the gradient \f$g_L = \nabla_xf(x) - \sum_i y_i \nabla_xc_i(x)
    - \sum_i z_i e_i\f$ of the Lagrangian function.
    The j-th component of gl, j = 0, ... ,  n-1, contains  \f$gl_j \f$.

 @param eval_fc is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_fc( ipc_ n, ipc_ m, const rpc_ x[], rpc_ f, rpc_ c[],
                     const void *userdata )
   \endcode
   The value of the objective function \f$f(x)\f$ and components of the
   constraint functions \f$c(x)\f$ evaluated at x=\f$x\f$
   must be assigned to f and c, and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_c via the structure \c userdata.

 @param eval_gj is a user-supplied function that must have the following
   signature:
   \code
      ipc_ eval_gj( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[],
                   rpc_ j[], const void *userdata )
   \endcode
   The components of the gradient \f$g = \nabla_x f(x\f$) and of the
   constraint Jacobian \f$J = \nabla_x c(x\f$) must  be assigned to g
   and j in the same order as presented to expo_import, and the
   function return value set to 0. If the evaluation is impossible at x,
   return should be set to a nonzero value.
   Data may be passed into \c eval_j via the structure \c userdata.

 @param eval_hl is a user-supplied function that must have the following
   signature:
   \code
        ipc_ eval_hl( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
                    rpc_ h[], const void *userdata )
   \endcode
   The nonzeros of the matrix \f$H = \nabla_{xx}f(x)\
   - \sum_{i=1}^m y_i  \nabla_{xx}c_i(x)\f$
   of the Hessian of the Lagrangian evaluated at x=\f$x\f$ and y=\f$y\f$ must
   be assigned to h in the same order as presented to expo_import, and the
   function return value set to 0. If the evaluation is impossible at (x,y),
   return should be set to a nonzero value.
   Data may be passed into \c eval_h via the structure \c userdata.

 */


// *-*-*-*-*-*-*-*-*-*-    E X P O  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void expo_information( void **data,
                      struct expo_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see expo_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    E X P O  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void expo_terminate( void **data,
                    struct expo_control_type *control,
                    struct expo_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see expo_control_type)

  @param[out] inform   is a struct containing output information
              (see expo_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example expot.c
   This is an example of how to use the package both when the matrices
   (Jacobian, Hessian) are directly available. Function call evaluations
   are illustrated. A variety of supported Hessian storage formats are shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false. In addition, see how
   parameters may be passed into the evaluation functions via \c userdata.\n

   \example expotf.c
   This is the same example, but now fortran-style indexing is used.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
