/** \file nls.h */

/*
 * THIS VERSION: GALAHAD 3.3 - 19/08/2021 AT 13:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 19th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package nls
 
  \section nls_intro Introduction

  \subsection nls_purpose Purpose

  This package uses a <b>regularization method to find a (local)
  unconstrained minimizer of a differentiable weighted sum-of-squares objective
  function
  \f[\mathbf{f(x) :=
   \frac{1}{2} \sum_{i=1}^m w_i^{} c_i^2(x) \equiv \frac{1}{2} \|c(x)\|^2_W}\f]
\manonly
  \n
  f(x):= 1/2 sum_{i=1}^m w_i c_i^2(x) = 1/2 ||c(x)||^2_W
  \n
\endmanonly
  of many variables \f$\mathbf{x}\f$ involving positive weights
  \f$\mathbf{w_i}\f$, \f$\mathbf{i=1,\ldots,m}\f$.</b>
  The method offers the choice of direct and iterative solution of the key
  regularization subproblems, and is most suitable for large problems.
  First derivatives of the <i>residual function</i>
  \f$c(x)\f$ are required, and if second derivatives of the
  \f$c_i(x)\f$ can be calculated, they may be exploited---if suitable products
  of the first or second derivatives with a vector may be found but not the
  derivatives themselves, that can also be used to advantage.

  \subsection nls_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection nls_date Originally released

  October 2016, C interface August 2021.

  \subsection nls_terminology Terminology

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

  \subsection nls_method Method

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
  (see the \ref nls_terminology section), and normally a two-norm 
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
  Newton-like process, by the GALAHAD packages RQS. The iterative approach
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

  \subsection nls_references Reference

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

  The specific methods employed here are discussed in

  N. I. M. Gould, J. A. Scott and T. Rees,
  ``Convergence and evaluation-complexity analysis of a regularized
  tensor-Newton method for solving nonlinear least-squares problems''.
  Computational Optimization and Applications 73(1) (2019) 1–35.

  \subsection nls_call_order Call order

  To solve a given problem, functions from the nls package must be called 
  in the following order:

  - \link nls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link nls_read_specfile \endlink (optional) - override control values 
      by reading replacement values from a file
  - \link nls_import \endlink - set up problem data structures and fixed
      values
  - \link nls_reset_control \endlink (optional) - possibly change control 
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of 
     - \link nls_solve_with_mat \endlink - solve using function calls to
       evaluate function, gradient and Hessian values
     - \link nls_solve_without_mat \endlink - solve using function calls to
       evaluate function and gradient values and Hessian-vector products
     - \link nls_solve_reverse_with_mat \endlink - solve returning to the
       calling program to obtain function, gradient and Hessian values, or
     - \link nls_solve_reverse_without_mat \endlink - solve returning to the
       calling prorgram to obtain function and gradient values and 
       Hessian-vector products
  - \link nls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link nls_terminate \endlink - deallocate data structures

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
  while A_ptr(m) holds the total number of entries plus one.
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
  while A_ptr(n) holds the total number of entries plus one.
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
  while H_ptr(n) holds the total number of entries plus one.
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
#endif

// include guard
#ifndef GALAHAD_NLS_H 
#define GALAHAD_NLS_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct nls_subproblem_control_type {

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief the level of output required. 
    /// \li \f$\leq\f$ 0 gives no output, 
    /// \li  = 1 gives a one-line summary for every iteration, 
    /// \li  = 2 gives a summary of the inner iteration for each iteration, 
    /// \li \f$\geq\f$ 3 gives increasingly verbose (debugging) output
    int print_level;

    /// \brief
    /// any printing will start on this iteration
    int start_print;

    /// \brief
    /// any printing will stop on this iteration
    int stop_print;

    /// \brief
    /// the number of iterations between printing
    int print_gap;

    /// \brief
    /// the maximum number of iterations performed
    int maxit;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// is the Jacobian matrix of first derivatives available (\f$\geq\f$ 2), 
    /// is access only via matrix-vector products (=1) or is it not available 
    /// (\f$\leq\f$ 0) ?
    int jacobian_available;

    /// \brief
    /// is the Hessian matrix of second derivatives available (\f$\geq\f$ 2), 
    /// is access only via matrix-vector products (=1) or is it not available 
    /// (\f$\leq\f$ 0) ?
    int hessian_available;

    /// \brief
    /// the model used. 
    ///
    /// Possible values are
    /// \li 0  dynamic (*not yet implemented*)
    /// \li 1  first-order (no Hessian)
    /// \li 2  barely second-order (identity Hessian)
    /// \li 3  Gauss-Newton (\f$J^T J\f$ Hessian)
    /// \li 4  second-order (exact Hessian)
    /// \li 5  Gauss-Newton to Newton transition
    /// \li 6  tensor Gauss-Newton treated as a least-squares model
    /// \li 7  tensor Gauss-Newton treated as a general model
    /// \li 8  tensor Gauss-Newton transition from a least-squares 
    ///        to a general mode
    int model;

    /// \brief 
    /// the regularization norm used. 

    /// The norm is defined via \f$\|v\|^2 = v^T S v\f$,
    /// and will define the preconditioner used for iterative methods.
    /// Possible values for \f$S\f$ are
    ///
    /// \li -3  user's own regularization norm
    /// \li -2  \f$S\f$ = limited-memory BFGS matrix (with
    ///         .PSLS_control.lbfgs_vectors history) (*not yet implemented*)
    /// \li -1  identity (= Euclidan two-norm)
    /// \li  0  automatic (*not yet implemented*)
    /// \li  1  diagonal, \f$S\f$ = diag( max( \f$J^TJ\f$ Hessian, 
    ///         .PSLS_contro.min_diagonal ) )
    /// \li  2  diagonal, \f$S\f$ = diag( max( Hessian, 
    ///         .PSLS_contro.min_diagonal ) )
    /// \li  3  banded, \f$S\f$ = band( Hessian ) with semi-bandwidth
    ///         .PSLS_control.semi_bandwidth
    /// \li  4  re-ordered band, P=band(order(A)) with semi-bandwidth
    ///         .PSLS_control.semi_bandwidth
    /// \li  5  full factorization, \f$S\f$ = Hessian, Schnabel-Eskow 
    ///         modification
    /// \li  6  full factorization, \f$S\f$ = Hessian, GMPS modification
    ///         (*not yet implemented*)
    /// \li  7  incomplete factorization of Hessian, Lin-More'
    /// \li  8  incomplete factorization of Hessian, HSL_MI28
    /// \li  9  incomplete factorization of Hessian, Munskgaard
    ///         (*not yet implemented*)
    /// \li 10  expanding band of Hessian (*not yet implemented*)
    int norm;

    /// \brief
    /// non-monotone \f$\leq\f$ 0 monotone strategy used, anything else 
    /// non-monotone strategy with this history length used
    int non_monotone;

    /// \brief
    /// define the weight-update strategy:
    /// 1 (basic), 2 (reset to zero when very successful),
    /// 3 (imitate TR), 4 (increase lower bound), 5 (GPT)
    int weight_update_strategy;

    /// \brief
    /// overall convergence tolerances. The iteration will terminate when
    /// \f$||c(x)||_2 \leq \f$ MAX( .stop_c_absolute, .stop_c_relative 
    ///   \f$ * \|c(x_{\mbox{initial}})\|_2\f$, or
    /// when the norm of the gradient, \f$g = J^T(x) c(x) / \|c(x)\|_2\f$, 
    /// of ||c||_2, satisfies
    /// \f$\|g\|_2 \leq\f$  MAX( .stop_g_absolute, .stop_g_relative 
    ///   \f$ * \|g_{\mbox{initial}}\|_2\f$, or
    /// if the step is less than .stop_s
    real_wp_ stop_c_absolute;
    /// see stop_c_absolute
    real_wp_ stop_c_relative;
    /// see stop_c_absolute
    real_wp_ stop_g_absolute;
    /// see stop_c_absolute
    real_wp_ stop_g_relative;
    /// see stop_c_absolute
    real_wp_ stop_s;

    /// \brief
    /// the regularization power (<2 => chosen according to the model)
    real_wp_ power;

    /// \brief
    /// initial value for the regularization weight (-ve => \f$1/\|g_0\|)\f$)
    real_wp_ initial_weight;

    /// \brief
    /// minimum permitted regularization weight
    real_wp_ minimum_weight;

    /// \brief
    /// initial value for the inner regularization weight for tensor GN 
    /// (-ve => 0)
    real_wp_ initial_inner_weight;

    /// \brief
    /// REAL ( KIND = wp ) :: initial_inner_weight = 0.0001_wp
    /// a potential iterate will only be accepted if the actual decrease
    /// f - f(x_new) is larger than .eta_successful times that predicted
    /// by a quadratic model of the decrease. The regularization weight will be
    /// decreaed if this relative decrease is greater than .eta_very_successful
    /// but smaller than .eta_too_successful
    real_wp_ eta_successful;
    /// see eta_successful
    real_wp_ eta_very_successful;
    /// see eta_successful
    real_wp_ eta_too_successful;

    /// \brief
    /// on very successful iterations, the regularization weight will be reduced
    /// by the factor .weight_decrease but no more than .weight_decrease_min
    /// while if the iteration is unsucceful, the weight will be increased by a
    /// factor .weight_increase but no more than .weight_increase_max
    /// (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
    /// and Toint, 2011)
    real_wp_ weight_decrease_min;

    /// \brief
    /// REAL ( KIND = wp ) :: weight_decrease = half
    real_wp_ weight_decrease;

    /// \brief
    /// REAL ( KIND = wp ) :: weight_increase = two
    real_wp_ weight_increase;
    /// see weight_increase
    real_wp_ weight_increase_max;

    /// \brief
    /// expert parameters as suggested in Gould, Porcelli and Toint, "Updating t
    /// regularization parameter in the adaptive cubic regularization algorithm"
    /// RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011),
    /// http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
    /// (these are denoted beta, epsilon_chi and alpha_max in the paper)
    real_wp_ reduce_gap;
    /// see reduce_gap
    real_wp_ tiny_gap;
    /// see reduce_gap
    real_wp_ large_root;

    /// \brief
    /// if the Gauss-Newto to Newton model is specified, switch to Newton as
    /// soon as the norm of the gradient g is smaller than switch_to_newton
    real_wp_ switch_to_newton;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    /// \brief
    /// use a direct (factorization) or (preconditioned) iterative method to
    /// find the search direction
    bool subproblem_direct;

    /// \brief
    /// should the weight be renormalized to account for a change in scaling?
    bool renormalize_weight;

    /// \brief
    /// allow the user to perform a "magic" step to improve the objective
    bool magic_step;

    /// \brief
    /// print values of the objective/gradient rather than ||c|| and its gradien
    bool print_obj;

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
    /// control parameters for RQS
    ///struct rqs_control_type rqs_control;

    /// \brief
    /// control parameters for GLRT
    ///struct glrt_control_type glrt_control;

    /// \brief
    /// control parameters for PSLS
    ///struct psls_control_type psls_control;

    /// \brief
    /// control parameters for BSC
    /// struct bsc_control_type bsc_control;

    /// \brief
    /// control parameters for ROOTS
    /// struct roots_control_type roots_control;
};

/**
 * control derived type as a C struct
 */
struct nls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief the level of output required. 
    /// \li \f$\leq\f$ 0 gives no output, 
    /// \li  = 1 gives a one-line summary for every iteration, 
    /// \li  = 2 gives a summary of the inner iteration for each iteration, 
    /// \li \f$\geq\f$ 3 gives increasingly verbose (debugging) output
    int print_level;

    /// \brief
    /// any printing will start on this iteration
    int start_print;

    /// \brief
    /// any printing will stop on this iteration
    int stop_print;

    /// \brief
    /// the number of iterations between printing
    int print_gap;

    /// \brief
    /// the maximum number of iterations performed
    int maxit;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// is the Jacobian matrix of first derivatives available (\f$\geq\f$ 2), 
    /// is access only via matrix-vector products (=1) or is it not 
    /// available (\f$\leq\f$ 0) ?
    int jacobian_available;

    /// \brief
    /// is the Hessian matrix of second derivatives available (\f$\geq\f$ 2), 
    /// is access only via matrix-vector products (=1) or is it not 
    /// available (\f$\leq\f$ 0) ?
    int hessian_available;

    /// \brief
    /// the model used. 
    ///
    /// Possible values are
    /// \li 0  dynamic (*not yet implemented*)
    /// \li 1  first-order (no Hessian)
    /// \li 2  barely second-order (identity Hessian)
    /// \li 3  Gauss-Newton (\f$J^T J\f$ Hessian)
    /// \li 4  second-order (exact Hessian)
    /// \li 5  Gauss-Newton to Newton transition
    /// \li 6  tensor Gauss-Newton treated as a least-squares model
    /// \li 7  tensor Gauss-Newton treated as a general model
    /// \li 8  tensor Gauss-Newton transition from a least-squares 
    ///        to a general mode
    int model;

    /// \brief 
    /// the regularization norm used. 

    /// The norm is defined via \f$\|v\|^2 = v^T S v\f$,
    /// and will define the preconditioner used for iterative methods.
    /// Possible values for \f$S\f$ are
    ///
    /// \li -3  user's own regularization norm
    /// \li -2  \f$S\f$ = limited-memory BFGS matrix (with
    ///         .PSLS_control.lbfgs_vectors history) (*not yet implemented*)
    /// \li -1  identity (= Euclidan two-norm)
    /// \li  0  automatic (*not yet implemented*)
    /// \li  1  diagonal, \f$S\f$ = diag( max( \f$J^TJ\f$ Hessian, 
    ///         .PSLS_contro.min_diagonal ) )
    /// \li  2  diagonal, \f$S\f$ = diag( max( Hessian, 
    ///         .PSLS_contro.min_diagonal ) )
    /// \li  3  banded, \f$S\f$ = band( Hessian ) with semi-bandwidth
    ///         .PSLS_control.semi_bandwidth
    /// \li  4  re-ordered band, P=band(order(A)) with semi-bandwidth
    ///         .PSLS_control.semi_bandwidth
    /// \li  5  full factorization, \f$S\f$ = Hessian, Schnabel-Eskow 
    ///         modification
    /// \li  6  full factorization, \f$S\f$ = Hessian, GMPS modification
    ///         (*not yet implemented*)
    /// \li  7  incomplete factorization of Hessian, Lin-More'
    /// \li  8  incomplete factorization of Hessian, HSL_MI28
    /// \li  9  incomplete factorization of Hessian, Munskgaard
    ///         (*not yet implemented*)
    /// \li 10  expanding band of Hessian (*not yet implemented*)
    int norm;

    /// \brief
    /// non-monotone \f$\leq\f$ 0 monotone strategy used, anything else 
    /// non-monotone strategy with this history length used
    int non_monotone;

    /// \brief
    /// define the weight-update strategy:
    /// 1 (basic), 2 (reset to zero when very successful),
    /// 3 (imitate TR), 4 (increase lower bound), 5 (GPT)
    int weight_update_strategy;

    /// \brief
    /// overall convergence tolerances. The iteration will terminate when
    /// \f$||c(x)||_2 \leq \f$ MAX( .stop_c_absolute, .stop_c_relative 
    ///   \f$ * \|c(x_{\mbox{initial}})\|_2\f$ or
    /// when the norm of the gradient, \f$g = J^T(x) c(x) / \|c(x)\|_2\f$,
    /// of ||c(x)||_2 satisfies
    /// \f$\|g\|_2 \leq\f$  MAX( .stop_g_absolute, .stop_g_relative 
    ///   \f$ * \|g_{\mbox{initial}}\|_2\f$, or
    /// if the step is less than .stop_s
    real_wp_ stop_c_absolute;
    /// see stop_c_absolute
    real_wp_ stop_c_relative;
    /// see stop_c_absolute
    real_wp_ stop_g_absolute;
    /// see stop_c_absolute
    real_wp_ stop_g_relative;
    /// see stop_c_absolute
    real_wp_ stop_s;

    /// \brief
    /// the regularization power (<2 => chosen according to the model)
    real_wp_ power;

    /// \brief
    /// initial value for the regularization weight (-ve => \f$1/\|g_0\|)\f$)
    real_wp_ initial_weight;

    /// \brief
    /// minimum permitted regularization weight
    real_wp_ minimum_weight;

    /// \brief
    /// initial value for the inner regularization weight for tensor GN 
    /// (-ve => 0)
    real_wp_ initial_inner_weight;

    /// \brief
    /// REAL ( KIND = wp ) :: initial_inner_weight = 0.0001_wp
    /// a potential iterate will only be accepted if the actual decrease
    /// f - f(x_new) is larger than .eta_successful times that predicted
    /// by a quadratic model of the decrease. The regularization weight will be
    /// decreaed if this relative decrease is greater than .eta_very_successful
    /// but smaller than .eta_too_successful
    real_wp_ eta_successful;
    /// see eta_successful
    real_wp_ eta_very_successful;
    /// see eta_successful
    real_wp_ eta_too_successful;

    /// \brief
    /// on very successful iterations, the regularization weight will be reduced
    /// by the factor .weight_decrease but no more than .weight_decrease_min
    /// while if the iteration is unsucceful, the weight will be increased by a
    /// factor .weight_increase but no more than .weight_increase_max
    /// (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
    /// and Toint, 2011)
    real_wp_ weight_decrease_min;

    /// \brief
    /// REAL ( KIND = wp ) :: weight_decrease = half
    real_wp_ weight_decrease;

    /// \brief
    /// REAL ( KIND = wp ) :: weight_increase = two
    real_wp_ weight_increase;
    /// see weight_increase
    real_wp_ weight_increase_max;

    /// \brief
    /// expert parameters as suggested in Gould, Porcelli and Toint, "Updating t
    /// regularization parameter in the adaptive cubic regularization algorithm"
    /// RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011),
    /// http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
    /// (these are denoted beta, epsilon_chi and alpha_max in the paper)
    real_wp_ reduce_gap;
    /// see reduce_gap
    real_wp_ tiny_gap;
    /// see reduce_gap
    real_wp_ large_root;

    /// \brief
    /// if the Gauss-Newto to Newton model is specified, switch to Newton as
    /// soon as the norm of the gradient g is smaller than switch_to_newton
    real_wp_ switch_to_newton;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    /// \brief
    /// use a direct (factorization) or (preconditioned) iterative method to
    /// find the search direction
    bool subproblem_direct;

    /// \brief
    /// should the weight be renormalized to account for a change in scaling?
    bool renormalize_weight;

    /// \brief
    /// allow the user to perform a "magic" step to improve the objective
    bool magic_step;

    /// \brief
    /// print values of the objective/gradient rather than ||c|| and its gradien
    bool print_obj;

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
    /// control parameters for the step-finding subproblem
    struct nls_subproblem_control_type subproblem_control;

    /// \brief
    /// control parameters for RQS
    ///struct rqs_control_type rqs_control;

    /// \brief
    /// control parameters for GLRT
    ///struct glrt_control_type glrt_control;

    /// \brief
    /// control parameters for PSLS
    ///struct psls_control_type psls_control;

    /// \brief
    /// control parameters for BSC
    /// struct bsc_control_type bsc_control;

    /// \brief
    /// control parameters for ROOTS
    /// struct roots_control_type roots_control;
};

/**
 * time derived type as a C struct
 */
struct nls_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    real_sp_ preprocess;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    real_sp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_sp_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    real_sp_ solve;

    /// \brief
    /// the total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    real_wp_ clock_preprocess;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
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
struct nls_subproblem_inform_type {

    /// \brief
    /// return status. See NLS_solve for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// the name of the user-supplied evaluation routine for which an error ocur
    char bad_eval[13];

    /// \brief
    /// the total number of iterations performed
    int iter;

    /// \brief
    /// the total number of CG iterations performed
    int cg_iter;

    /// \brief
    /// the total number of evaluations of the residual function c(x)
    int c_eval;

    /// \brief
    /// the total number of evaluations of the Jacobian J(x) of c(x)
    int j_eval;

    /// \brief
    /// the total number of evaluations of the scaled Hessian H(x,y) of c(x)
    int h_eval;

    /// \brief
    /// the maximum number of factorizations in a sub-problem solve
    int factorization_max;

    /// \brief
    /// the return status from the factorization
    int factorization_status;

    /// \brief
    /// the maximum number of entries in the factors
    int max_entries_factors;

    /// \brief
    /// the total integer workspace required for the factorization
    int factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int factorization_real;

    /// \brief
    /// the average number of factorizations per sub-problem solve
    real_wp_ factorization_average;

    /// \brief
    /// the value of the objective function \f$\frac{1}{2}\|c(x)\|^2_W\f$ 
    /// at the best estimate the solution, x, determined by NLS_solve
    real_wp_ obj;

    /// \brief
    /// the norm of the residual \f$\|c(x)\|_W\f$ at the best estimate of 
    /// the solution x, determined by NLS_solve
    real_wp_ norm_c;

    /// \brief
    /// the norm of the gradient of \f$\|c(x)\|_W\f$ of the objective function
    /// at the best estimate, x, of the solution determined by NLS_solve
    real_wp_ norm_g;

    /// \brief
    /// the final regularization weight used
    real_wp_ weight;

    /// \brief
    /// timings (see above)
    struct nls_time_type time;

    /// \brief
    /// inform parameters for RQS
    /// struct rqs_inform_type rqs_inform;

    /// \brief
    /// inform parameters for GLRT
    /// struct glrt_inform_type glrt_inform;

    /// \brief
    /// inform parameters for PSLS
    /// struct psls_inform_type psls_inform;

    /// \brief
    /// inform parameters for BSC
    /// struct bsc_inform_type bsc_inform;

    /// \brief
    /// inform parameters for ROOTS
    /// struct roots_inform_type roots_inform;
};

/**
 * inform derived type as a C struct
 */
struct nls_inform_type {

    /// \brief
    /// return status. See NLS_solve for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// the name of the user-supplied evaluation routine for which an error ocur
    char bad_eval[13];

    /// \brief
    /// the total number of iterations performed
    int iter;

    /// \brief
    /// the total number of CG iterations performed
    int cg_iter;

    /// \brief
    /// the total number of evaluations of the residual function c(x)
    int c_eval;

    /// \brief
    /// the total number of evaluations of the Jacobian J(x) of c(x)
    int j_eval;

    /// \brief
    /// the total number of evaluations of the scaled Hessian H(x,y) of c(x)
    int h_eval;

    /// \brief
    /// the maximum number of factorizations in a sub-problem solve
    int factorization_max;

    /// \brief
    /// the return status from the factorization
    int factorization_status;

    /// \brief
    /// the maximum number of entries in the factors
    int max_entries_factors;

    /// \brief
    /// the total integer workspace required for the factorization
    int factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int factorization_real;

    /// \brief
    /// the average number of factorizations per sub-problem solve
    real_wp_ factorization_average;

    /// \brief
    /// the value of the objective function \f$\frac{1}{2}\|c(x)\|^2_W\f$ 
    /// at the best estimate the solution, x, determined by NLS_solve
    real_wp_ obj;

    /// \brief
    /// the norm of the residual \f$\|c(x)\|_W\f$ at the best estimate of 
    /// the solution x, determined by NLS_solve
    real_wp_ norm_c;

    /// \brief
    /// the norm of the gradient of \f$\|c(x)\|_W\f$ of the objective function
    /// at the best estimate, x, of the solution determined by NLS_solve
    real_wp_ norm_g;

    /// \brief
    /// the final regularization weight used
    real_wp_ weight;

    /// \brief
    /// timings (see above)
    struct nls_time_type time;

    /// \brief
    /// inform parameters for subproblem
    struct nls_subproblem_inform_type subproblem_inform;

    /// \brief
    /// inform parameters for RQS
    /// struct rqs_inform_type rqs_inform;

    /// \brief
    /// inform parameters for GLRT
    /// struct glrt_inform_type glrt_inform;

    /// \brief
    /// inform parameters for PSLS
    /// struct psls_inform_type psls_inform;

    /// \brief
    /// inform parameters for BSC
    /// struct bsc_inform_type bsc_inform;

    /// \brief
    /// inform parameters for ROOTS
    /// struct roots_inform_type roots_inform;
};

// *-*-*-*-*-*-*-*-*-*-    N L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void nls_initialize( void **data, 
                     struct nls_control_type *control,
                     struct nls_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information 
              (see nls_control_type)
  @param[out] inform   is a struct containing output information
              (see nls_inform_type) 
*/

// *-*-*-*-*-*-*-*-*-    N L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void nls_read_specfile( struct nls_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control  is a struct containing control information 
              (see nls_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    N L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void nls_import( struct nls_control_type *control,
                 void **data,
                 int *status, 
                 int n,
                 int m,
                 const char J_type[], 
                 int J_ne, 
                 const int J_row[],
                 const int J_col[], 
                 const int J_ptr[],
                 const char H_type[], 
                 int H_ne, 
                 const int H_row[],
                 const int H_col[], 
                 const int H_ptr[],
                 const char P_type[], 
                 int P_ne, 
                 const int P_row[],
                 const int P_col[], 
                 const int P_ptr[],
                 const real_wp_ w[] );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see nls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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

 @param[in] n is a scalar variable of type int, that holds the number of
    variables.

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in]  J_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices symmetric storage scheme \endlink 
   used for the Jacobian, \f$J\f$. It should be one of 'coordinate', 
  'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian
  is via matrix-vector products; lower or upper case variants are allowed.

 @param[in]  J_ne is a scalar variable of type int, that holds the number of
   entries in \f$J\f$ in the sparse co-ordinate storage scheme. 
   It need not be set for any of the other schemes.

 @param[in]  J_row is a one-dimensional array of size J_ne and type int, that 
   holds the row indices of \f$J\f$ in the sparse co-ordinate storage scheme. 
   It need not be set for any of the other schemes, 
   and in this case can be NULL.

 @param[in]  J_col is a one-dimensional array of size J_ne and type int,
   that holds the column indices of \f$J\f$ in either the sparse co-ordinate, 
   or the sparse row-wise storage scheme. It need not be set when the 
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  J_ptr is a one-dimensional array of size n+1 and type int,
   that holds the starting position of each row of \f$J\f$, as well as the 
   total number of entries plus one, in the sparse row-wise storage scheme. 
   It need not be set when the other schemes are used, 
   and in this case can be NULL.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink 
   used for the Hessian, \f$H\f$. It should be one of 'coordinate', 
   'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to 
   \f$H\f$ is via matrix-vector products; lower or upper case variants 
    are allowed.

 @param[in]  H_ne is a scalar variable of type int, that holds the number of
   entries in the lower triangular part of \f$H\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other three schemes.

 @param[in]  H_row is a one-dimensional array of size H_ne and type int, that 
   holds the row indices of the lower triangular part of \f$H\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  H_col is a one-dimensional array of size H_ne and type int,
   that holds the column indices of the lower triangular part of \f$H\f$ in 
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It 
   need not be set when the dense or diagonal storage schemes are used, 
   and in this case can be NULL.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type int,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$, as well as the total number of entries plus one,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

 @param[in]  P_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices symmetric storage scheme \endlink 
   used for the residual-Hessians-vector product matrix, \f$P\f$. It should be 
   one of 'coordinate', 'sparse_by_columns', 'dense_by_columns' or 'absent', 
   the latter if access to \f$P\f$ is  via matrix-vector products; 
   lower or upper case variants are allowed.

 @param[in]  P_ne is a scalar variable of type int, that holds the number of
   entries in \f$P\f$ in the sparse co-ordinate storage scheme. 
   It need not be set for any of the other schemes.

 @param[in]  P_row is a one-dimensional array of size P_ne and type int, that 
   holds the row indices of \f$P\f$ in either the sparse co-ordinate, 
   or the sparse column-wise storage scheme. 
   It need not be set when the dense storage scheme is used, 
   and in this case can be NULL.

 @param[in]  P_col is a one-dimensional array of size P_ne and type int, that
   holds the row indices of \f$P\f$ in the sparse co-ordinate storage scheme. 
   It need not be set for any of the other schemes, 
   and in this case can be NULL.

 @param[in]  P_ptr is a one-dimensional array of size m+1 and type int,
   that holds the starting position of each row of \f$P\f$, as well as the 
   total number of entries plus one, in the sparse row-wise storage scheme. 
   It need not be set when the other schemes are used, 
   and in this case can be NULL.

 @param[in] w is a one-dimensional array of size m and type double, 
   that holds the values \f$w\f$ of the weights on the residuals in the
   least-squares objective function. It need not be set if the weights are
   all ones, and in this case can be NULL
 */

//  *-*-*-*-*-*-*-*-*-   N L S _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void nls_reset_control( struct nls_control_type *control,
                        void **data,
                        int *status, );

/*!< 
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see nls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
 */

//  *-*-*-*-*-*-*-*-*-   N L S _ S O L V E _ W I T H _ M A T   -*-*-*-*-*-*-*-*

void nls_solve_with_mat( void **data,
                         void *userdata, 
                         int *status, 
                         int n, 
                         int m, 
                         real_wp_ x[], 
                         real_wp_ c[],
                         real_wp_ g[],
                         int (*eval_c)(
                           int, int, const real_wp_[], real_wp_[], 
                           const void * ), 
                         int j_ne, 
                         int (*eval_j)(
                           int, int, int, const real_wp_[], real_wp_[],
                           const void * ),
                         int h_ne, 
                         int (*eval_h)(
                           int, int, int, const real_wp_[], const real_wp_[], 
                           real_wp_[], const void * ),
                         int p_ne, 
                         int (*eval_hprods)(
                           int, int, int, const real_wp_[], 
                           const real_wp_[], real_wp_[], bool, 
                           const void * ) );

/*!<
 Find a local minimizer of a given function using a trust-region method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is 
 provided specifically, and all function/derivative information is 
 available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type int, that gives
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
 
 @param[in] n is a scalar variable of type int, that holds the number of
    variables.

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[out] c is a one-dimensional array of size m and type double, that 
    holds the residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.
  
 @param[out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param eval_c is a user-supplied function that must have the following 
   signature:
   \code
        int eval_c( int n, const double x[], double c[], const void *userdata ) 
   \endcode
   The componnts of the residual function \f$c(x)\f$ evaluated at x=\f$x\f$ 
   must be assigned to c, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_c via the structure \c userdata. 

 @param[in] j_ne is a scalar variable of type int, that holds the number of 
    entries in the Jacobian matrix \f$J\f$.

 @param eval_j is a user-supplied function that must have the following 
   signature:
   \code
      int eval_j( int n, int m, int jne, const double x[], double j[], 
                  const void *userdata )
   \endcode
   The components of the Jacobian \f$J = \nabla_x c(x\f$) of the residuals must
   be assigned to j in the same order as presented to nls_import, and the 
   function return value set to 0. If the evaluation is impossible at x, 
   return should be set to a nonzero value.
   Data may be passed into \c eval_j via the structure \c userdata. 
 
 @param[in] h_ne is a scalar variable of type int, that holds the number of 
    entries in the lower triangular part of the Hessian matrix \f$H\f$
    if it is used.

 @param eval_h is a user-supplied function that must have the following 
   signature:
   \code
        int eval_h( int n, int m, int hne, const double x[], const double y[], 
                    double h[], const void *userdata )
   \endcode
   The nonzeros of the matrix \f$H = \sum_{i=1}^m y_i  \nabla_{xx}c_i(x)\f$ 
   of the weighted residual Hessian evaluated at x=\f$x\f$ and y=\f$y\f$ must 
   be assigned to h in the same order as presented to nls_import, and the 
   function return value set to 0. If the evaluation is impossible at x, 
   return should be set to a nonzero value.
   Data may be passed into \c eval_h via the structure \c userdata. 
 
 @param[in] p_ne is a scalar variable of type int, that holds the number of 
    entries in the residual-Hessians-vector product matrix \f$P\f$ if it 
    is used.

 @param  eval_hprods is an optional user-supplied function that may be NULL. 
   If non-NULL, it must have the following signature:
   \code

       int eval_hprods( int n, int m, int pne, const double x[], 
                           const double v[], double p[], bool got_h, 
                           const void *userdata ) );

   \endcode
   The entries of the matrix \f$P\f$, whose i-th column is the 
   product \f$\nabla_{xx}c_i(x) v\f$ between \f$\nabla_{xx}c_i(x)\f$, the 
   Hessian of the i-th component of the residual \f$c(x)\f$ at x=\f$x\f$, and 
   v=\f$v\f$ must be returned in p and the function return value set to 0. 
   If the evaluation is impossible at x, return should be set to a nonzero 
   value.   Data may be passed into \c eval_hprods via the structure 
   \c userdata. 
 */ 

//  *-*-*-*-*-*-*-*-   N L S _ S O L V E _ W I T H O U T _ M A T   -*-*-*-*-*-*

void nls_solve_without_mat( void **data,
                            void *userdata, 
                            int *status, 
                            int n, 
                            int m, 
                            real_wp_ x[], 
                            real_wp_ c[], 
                            real_wp_ g[], 
                            int (*eval_c)(
                              int, int, const real_wp_[], real_wp_[], 
                              const void * ), 
                            int (*eval_jprod)(
                              int, int, const real_wp_[], const bool, 
                              real_wp_[], const real_wp_[], bool, 
                              const void * ),
                            int (*eval_hprod)(
                              int, int, const real_wp_[], const real_wp_[], 
                              real_wp_[], const real_wp_[], bool, 
                              const void * ), 
                            int p_ne, 
                            int (*eval_hprods)(
                              int, int, int, const real_wp_[], 
                              const real_wp_[], real_wp_[], bool, 
                              const void * ) );

/*!<
 Find a local minimizer of a given function using a trust-region method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is 
 provided by Hessian-vector products, and all function/derivative 
 information is available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type int, that gives
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
 
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[out] c is a one-dimensional array of size m and type double, that 
    holds the residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.
  
 @param[out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param eval_c is a user-supplied function that must have the following 
   signature:
   \code
        int eval_c( int n, const double x[], double c[], const void *userdata ) 
   \endcode
   The componnts of the residual function \f$c(x)\f$ evaluated at x=\f$x\f$ 
   must be assigned to c, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_c via the structure \c userdata. 

 @param eval_jprod is a user-supplied function that must have the following 
   signature:
   \code
      int eval_jprod( int n, int m, const double x[], bool transpose, 
                      double u[], const double v[], bool got_j, 
                      const void *userdata )
   \endcode
   The sum \f$u + \nabla_{x}c_(x) v\f$ (if tranpose is false) or
   The sum \f$u + (\nabla_{x}c_(x))^T v\f$ (if tranpose is true)
   bewteen the product of the Jacobian \f$\nabla_{x}c_(x)\f$ or its tranpose
   with the vector v=\f$v\f$ and the vector $\f$u\f$ must be returned in u, 
   and the function return value set to 0. If the evaluation is impossible 
   at x, return should be set to a nonzero value.
   Data may be passed into \c eval_jprod via the structure \c userdata. 

 @param eval_hprod is a user-supplied function that must have the following 
   signature:
   \code
        int eval_hprod( int n, int m, const double x[], const double y[], 
                        double u[], const double v[], bool got_h, 
                        const void *userdata )
   \endcode
   The sum \f$u + \sum_{i=1}^m y_i  \nabla_{xx}c_i(x) v\f$ of the product of 
   the weighted residual Hessian \f$H = \sum_{i=1}^m y_i  \nabla_{xx}c_i(x)\f$ 
   evaluated at x=\f$x\f$ and y=\f$y\f$ with the vector v=\f$v\f$ and the 
   vector $\f$u\f$ must be returned in u, and the function return value 
   set to 0. If the evaluation is impossible at x, return should be set to 
   a nonzero value.
   The Hessians have already been evaluated or used at x if got_h is true.
   Data may be passed into \c eval_hprod via the structure \c userdata. 

 @param[in] p_ne is a scalar variable of type int, that holds the number of 
    entries in the residual-Hessians-vector product matrix \f$P\f$ if it 
    is used.

 @param  eval_hprods is an optional user-supplied function that may be NULL. 
   If non-NULL, it must have the following signature:
   \code
       int eval_hprods( int n, int m, int p_ne, const double x[], 
                        const double v[], double pval[], bool got_h, 
                        const void *userdata )
   \endcode
   The entries of the matrix \f$P\f$, whose i-th column is the 
   product \f$\nabla_{xx}c_i(x) v\f$ between \f$\nabla_{xx}c_i(x)\f$, the 
   Hessian of the i-th component of the residual \f$c(x)\f$ at x=\f$x\f$, and 
   v=\f$v\f$ must be returned in pval and the function return value set to 0. 
   If the evaluation is impossible at x, return should be set to a nonzero 
   value.   Data may be passed into \c eval_hprods via the structure 
   \c userdata. 
 */ 

//  *-*-*-*-*-   N L S _ S O L V E _ R E V E R S E _ W I T H _ M A T   -*-*-*-*

void nls_solve_reverse_with_mat( void **data,
                                 int *status, 
                                 int *eval_status, 
                                 int n, 
                                 int m, 
                                 real_wp_ x[], 
                                 real_wp_ c[], 
                                 real_wp_ g[], 
                                 int j_ne, 
                                 real_wp_ J_val[], 
                                 const real_wp_ y[], 
                                 int h_ne, 
                                 real_wp_ H_val[], 
                                 real_wp_ v[],
                                 int p_ne, 
                                 real_wp_ P_val[] );

/*!<
 Find a local minimizer of a given function using a trust-region method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is 
 provided specifically, but function/derivative information is only 
 available by returning to the calling procedure

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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

  \li  2. The user should compute the vector of residuals \f$c(x)\f$ at 
         the point \f$x\f$ indicated in x and then re-enter the function. 
         The required value should be set in c, and eval_status should be 
         set to 0. If the user is unable to evaluate \f$c(x)\f$--- for 
         instance, if the function is undefined at \f$x\f$--- the user need 
         not set c, but should then set eval_status to a non-zero value.

  \li   3. The user should compute the Jacobian of the vector of residual 
         functions, \f$\nabla_x c(x)\f$, at the point \f$x\f$ indicated in x 
         and then re-enter the function. The l-th component of the Jacobian 
         stored according to the scheme specified for the remainder of 
         \f$J\f$ in the earlier call to nls_import should be set in J_val[l], 
         for l = 0, ..., J_ne-1 and eval_status should be set to 0. 
         If the user is unable to evaluate a component of \f$J\f$ --- for 
         instance, if a component of the matrix is undefined at 
         \f$x\f$ --- the user need not set J_val, but should 
         then set eval_status to a non-zero value.

 @param status (continued)
  \li   4. The user should compute the matrix
         \f$H = \sum_{i=1}^m v_i  \nabla_{xx}c_i(x)\f$ 
         of weighted residual Hessian evaluated at x=\f$x\f$ and v=\f$v\f$
         and then re-enter the function. The l-th component of the matrix 
         stored according to the scheme specified for the remainder of 
         \f$H\f$ in the earlier call to nls_import should be set in H_val[l], 
         for l = 0, ..., H_ne-1 and eval_status should be set to 0. 
         If the user is unable to evaluate a component of \f$H\f$ --- for 
         instance, if a component of the matrix is undefined at 
         \f$x\f$ --- the user need not set H_val, but should 
         then set eval_status to a non-zero value. \b Note that this
         return will not happen if the Gauss-Newton model is selected.
  \li   7. The user should compute the entries of the matrix \f$P\f$, 
          whose i-th column is the product \f$\nabla_{xx}c_i(x) v\f$ between 
         \f$\nabla_{xx}c_i(x)\f$, the Hessian of the i-th component of the 
         residual \f$c(x)\f$ at x=\f$x\f$, and v=\f$v\f$ and then re-enter 
         the function. The l-th component of the matrix 
         stored according to the scheme specified for the remainder of 
         \f$P\f$ in the earlier call to nls_import should be set in P_val[l], 
         for l = 0, ..., P_ne-1 and eval_status should be set to 0. 
         If the user is unable to evaluate a component of \f$P\f$ --- for 
         instance, if a component of the matrix is undefined at 
         \f$x\f$ --- the user need not set P_val, but should 
         then set eval_status to a non-zero value. 
         \b Note that this return will not happen if either the Gauss-Newton 
         or Newton models is selected.
 
 @param[in,out] eval_status is a scalar variable of type int, that is used to 
    indicate if  objective function/gradient/Hessian values can be provided 
    (see above) 
  
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in,out] c is a one-dimensional array of size m and type double, that 
    holds the residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.
    See status = 2, above, for more details.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param[in] j_ne is a scalar variable of type int, that holds the number of 
    entries in the Jacobian matrix \f$J\f$.
 
 @param[in] J_val is a one-dimensional array of size ne and type double, 
    that holds the values of the entries of the Jacobian matrix \f$J\f$ 
    in any of the available storage schemes. 
    See status = 3, above, for more details.

 @param[in,out] y is a one-dimensional array of size m and type double, that is 
    used for reverse communication.
    See status = 4 above for more details.

 @param[in] h_ne is a scalar variable of type int, that holds the number of 
    entries in the lower triangular part of the Hessian matrix \f$H\f$.
 
 @param[in] H_val is a one-dimensional array of size ne and type double, 
    that holds the values of the entries of the lower triangular part of the 
    Hessian matrix \f$H\f$ in any of the available storage schemes.
    See status = 4, above, for more details.

 @param[in,out] v is a one-dimensional array of size n and type double, that is 
    used for reverse communication.
    See status = 7, above, for more details.

 @param[in] p_ne is a scalar variable of type int, that holds the number of 
    entries in the residual-Hessians-vector product matrix, \f$P\f$.
 
 @param[in] P_val is a one-dimensional array of size ne and type double, 
    that holds the values of the entries of the residual-Hessians-vector 
    product matrix, \f$P\f$.
    See status = 7, above, for more details.
*/  

//  *-*-*-   N L S _ S O L V E _ R E V E R S E _ W I T H O U T _ M A T   -*-*-*

void nls_solve_reverse_without_mat( void **data,
                                    int *status, 
                                    int *eval_status, 
                                    int n, 
                                    int m, 
                                    real_wp_ x[], 
                                    real_wp_ c[], 
                                    real_wp_ g[], 
                                    bool *transpose,
                                    real_wp_ u[], 
                                    real_wp_ v[],
                                    real_wp_ y[],
                                    int p_ne, 
                                    real_wp_ P_val[] );

/*!<
 Find a local minimizer of a given function using a trust-region method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is 
 provided by Hessian-vector products, but function/derivative information 
 is only available by returning to the calling procedure.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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

 @param status (continued)
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

  \li  2. The user should compute the vector of residuals \f$c(x)\f$ at 
         the point \f$x\f$ indicated in x and then re-enter the function. 
         The required value should be set in c, and eval_status should be 
         set to 0. If the user is unable to evaluate \f$c(x)\f$--- for 
         instance, if the function is undefined at \f$x\f$--- the user need 
         not set c, but should then set eval_status to a non-zero value.

  \li  5. The user should compute the sum \f$u + \nabla_{x}c_(x) v\f$ 
         (if tranpose is false) or \f$u + (\nabla_{x}c_(x))^T v\f$ 
         (if tranpose is true) between the product of the Jacobian 
         \f$\nabla_{x}c_(x)\f$ or its tranpose with the vector v=\f$v\f$ and 
         the vector u = $\f$u\f$, and then re-enter the function.
         The result should be set in u, and eval_status should be set to 0. 
         If the user is unable to evaluate the sum --- for 
         instance, if the Jacobian is undefined at \f$x\f$ --- the user need 
         not set u, but should then set eval_status to a non-zero value.

  \li  6. The user should compute the sum 
         \f$u + \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v\f$ between the product 
         of the weighted residual Hessian 
         \f$H = \sum_{i=1}^m y_i  \nabla_{xx}c_i(x)\f$ 
         evaluated at x=\f$x\f$ and y=\f$y\f$ with the vector v=\f$v\f$
         and the the vector u = $\f$u\f$, and then re-enter the function.
         The result should be set in u, and eval_status should be set to 0. 
         If the user is unable to evaluate the sum --- for 
         instance, if the weifghted residual Hessian is undefined at 
         \f$x\f$ --- the user need not set u, but should then set eval_status 
         to a non-zero value.

  \li   7. The user should compute the entries of the matrix \f$P\f$, whose 
         i-th column is the product \f$\nabla_{xx}c_i(x) v\f$ between 
         \f$\nabla_{xx}c_i(x)\f$, the Hessian of the i-th component of the 
         residual \f$c(x)\f$ at x=\f$x\f$, and v=\f$v\f$ and then re-enter 
         the function. The l-th component of the matrix 
         stored according to the scheme specified for the remainder of 
         \f$P\f$ in the earlier call to nls_import should be set in P_val[l], 
         for l = 0, ..., P_ne-1 and eval_status should be set to 0. 
         If the user is unable to evaluate a component of \f$P\f$ --- for 
         instance, if a component of the matrix is undefined at 
         \f$x\f$ --- the user need not set P_val, but should 
         then set eval_status to a non-zero value.
         \b Note that this return will not happen if either the Gauss-Newton 
         or Newton models is selected.
 
 @param[in,out] eval_status is a scalar variable of type int, that is used to 
    indicate if  objective function/gradient/Hessian values can be provided 
    (see above) 
  
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in] m is a scalar variable of type int, that holds the number of
    residuals.

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in,out] c is a one-dimensional array of size m and type double, that 
    holds the residual \f$c(x)\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.
    See status = 2, above, for more details.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param[out] transpose is a scalar variable of type bool, that indicates
   whether the product with Jacobian or its transpose should be obtained when 
   status=5.

 @param[in,out] u is a one-dimensional array of size max(n,m) and type double, 
    that is used for reverse communication.
    See status = 5,6 above for more details.
  
 @param[in,out] v is a one-dimensional array of size max(n,m) and type double, 
    that is used for reverse communication.
    See status = 5,6,7 above for more details.

 @param[in,out] y is a one-dimensional array of size m and type double, that is 
    used for reverse communication.
    See status = 6 above for more details.

 @param[in] p_ne is a scalar variable of type int, that holds the number of 
    entries in the residual-Hessians-vector product matrix, \f$P\f$.
 
 @param[in] P_val is a one-dimensional array of size P_ne and type double, 
    that holds the values of the entries of the residual-Hessians-vector 
    product matrix, \f$P\f$.
    See status = 7, above, for more details.

*/  

// *-*-*-*-*-*-*-*-*-*-    N L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void nls_information( void **data,
                      struct nls_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see nls_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    N L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void nls_terminate( void **data, 
                    struct nls_control_type *control, 
                    struct nls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see nls_control_type)

  @param[out] inform   is a struct containing output information
              (see nls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example nlst.c
   This is an example of how to use the package both when the matrices
   (Jacobian, Hessian, residual-Hessians-vector product)
   are directly available and when their product with vectors may be found.
   Both function call evaluations and returns to the calling program
   to find the required values are illustrated. A variety of supported 
   Hessian storage formats are shown.
  
   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false. In addition, see how 
   parameters may be passed into the evaluation functions via \c userdata.\n

   \example nlstf.c
   This is the same example, but now fortran-style indexing is used.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
