//* \file galahad_qpa.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_QPA C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package qpa

  \section qpa_intro Introduction

  \subsection qpa_purpose Purpose

This package uses a working-set method
to solve the <b>\f$\ell_1\f$ quadratic programming problem</b>
\latexonly
  \[
   (1)\;\;\minin{x \in \smallRe^n} q(x) + \rho_g v_g(x) +\rho_b v_b(x)
  \]
\endlatexonly
\htmlonly
  $$
(1)\;\;\min_{x \in R^n} q(x) + \rho_g v_g(x) +\rho_b v_b(x)
  $$
\endhtmlonly
\manonly
  \n
  (1) min_{x in R^n} q(x) + rho_g v_g(x) + rho_b v_b(x)
  \n
\endmanonly
involving the quadratic objective
\latexonly
  \[
  q(x) = \frac{1}{2} x^T H x + g^T x + f
  \]
\endlatexonly
\htmlonly
  $$
  q(x) = \frac{1}{2} x^T H x + g^T x + f
  $$
\endhtmlonly
\manonly
  \n
  q(x) = 1/2 x^T H x + g^T x + f
  \n
\endmanonly
and the infeasibilities
\latexonly
  \[
v_g(x) = \sum_{i=1}^{m} \max ( c_i^l - a_i^T x, 0 ) + \sum_{i=1}^{m} \max ( a_i^T x - c_i^u, 0 )
  \]
\endlatexonly
\htmlonly
  $$
v_g(x) = \sum_{i=1}^{m} \max ( c_i^l - a_i^T x, 0 ) + \sum_{i=1}^{m} \max ( a_i^T x - c_i^u, 0 )
  $$
\endhtmlonly
\manonly
  \n
   v_g(x) = sum_{i=1}^m max(c_i^l - a_i^T x, 0) +
            sum_{i=1}^m max(a_i^T x - c_i^u, 0)
  \n
\endmanonly
and
\latexonly
  \[
v_b(x) = \sum_{j=1}^{n} \max ( x_j^l - x_j, 0 )
 + \sum_{j=1}^{n} \max ( x_j  - x_j^u , 0 ),
  \]
\endlatexonly
\htmlonly
  $$
v_b(x) = \sum_{j=1}^{n} \max ( x_j^l - x_j, 0 )
 + \sum_{j=1}^{n} \max ( x_j  - x_j^u , 0 ),
  $$
\endhtmlonly
\manonly
  \n
   v_b(x) = sum_{j=1}^{n} max(x_j^l - x_j, 0) +
            sum_{j=1}^{n} max(x_j - x_j^u, 0),
  \n
\endmanonly
  where the \f$n\f$ by \f$n\f$ symmetric matrix \f$H\f$,
  the vectors \f$g\f$, \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$H\f$
  or the matrix \f$A\f$ of vectors \f$a_i\f$.

  The package may also be used to solve the <b>quadratic programming problem</b>
  \f[(2) \;\; \mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  (2)   minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  subject to the general linear constraints
  \f[(3) \;\; c_i^l  \leq  a_i^Tx  \leq c_i^u, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
   (3)  c_i^l \[<=] a_i^Tx \[<=] c_i^u, i = 1, ... , m,
  \n
\endmanonly
  and the simple bound constraints
  \f[(4) \;\; x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   (4)   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  by automatically adjusting \f$\rho_b\f$ in (1).

  Similarly, the package is capable of solving the
  <b>bound-constrained \f$\ell_1\f$ quadratic programming problem</b>
\latexonly
  \[
  (5) \;\; \minin{x \in \smallRe^n} q(x) + \rho_g v_g(x),
  \]
\endlatexonly
\htmlonly
  $$
  (5) \;\; \min{x \in R^n} q(x) + \rho_g v_g(x),
  $$
\endhtmlonly

\manonly
  \n
  (5) min_{x in R^n} q(x) + rho_g v_g(x),
  \n
\endmanonly
  subject to the simple bound constraints (4),
  by automatically adjusting \f$\rho_b\f$ in (1).

  If the matrix \f$H\f$ is positive semi-definite, a global
  solution is found. However, if \f$H\f$ is indefinite,
  the procedure may find a (weak second-order) critical point
  that is not the global solution to the given problem.

  <b>N.B.</b> In many cases, the alternative GALAHAD quadratic
  programming package QPB is faster, and thus to be preferred.

  \subsection qpa_authors Authors

  N. I. M. Gould and D. P. Robinson, STFC-Rutherford Appleton Laboratory,
  England, and Philippe L. Toint, University of Namur, Belgium.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection qpa_date Originally released

  October 2001, C interface January 2022.

  \subsection qpa_terminology Terminology

  The required solution \f$x\f$ to (2)-(4) necessarily satisfies
  the primal optimality conditions
  \f[\mbox{(1a) $\hspace{66mm} A x = c\hspace{66mm}$}\f]
\manonly
  \n
  A x = c
  \n
\endmanonly
  and
  \f[\mbox{$\hspace{52mm} c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u,\hspace{52mm}$} \f]
\manonly
  \n
  c^l \[<=] c \[<=] c^u, x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[\mbox{$\hspace{58mm} H x + g = A^T y + z\hspace{58mm}$}\f]
\manonly
  \n
  H x + g = A^T y + z
  \n
\endmanonly
  where
  \f[\mbox{$\hspace{24mm} y = y^l + y^u, \;\; z = z^l + z^u, \,\,
   y^l \geq 0 , \;\;  y^u \leq 0 , \;\;
   z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,\hspace{24mm}$} \f]
\manonly
  \n
   y = y^l + y^u, z = z^l + z^u, y^l \[>=] 0, y^u \[<=] 0,
       z^l \[>=] 0 and z^u \[<=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[\mbox{$\hspace{12mm}
  ( A x - c^l )^T y^l = 0  ,\;\;  ( A x - c^u )^T y^u = 0  ,\;\;
  (x -x^l )^T z^l = 0 \;\;  \mbox{and} \;\; (x -x^u )^T z^u = 0,\hspace{12mm} $}\f]
\manonly
  \n
      (A x - c^l)^T y^l = 0, (A x - c^u)^T y^u = 0,
      (x -x^l)^T z^l = 0 and (x -x^u)^T z^u = 0,
  \n
\endmanonly
  where the vectors \f$y\f$ and \f$z\f$ are known as the Lagrange multipliers
  for the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection qpb_method Method

  At the \f$k\f$-th iteration of the method, an improvement to the value
  of the merit function
  \f$m(x, \rho_g, \rho_b ) = q(x) + \rho_g v_g(x) + \rho_b v_b(x)\f$ at
  \f$x = x^{(k)}\f$ is sought. This is achieved by
  first computing a search direction \f$s^{(k)}\f$, and then setting
  \f$x^{(k+1)} = x^{(k)} + \alpha^{(k)} s^{(k)}\f$, where the stepsize
  \f$\alpha^{(k)}\f$ is chosen as the first local minimizer of
  \f$\phi(\alpha) = m( x^{(k)} + \alpha s^{(k)} , \rho_g, \rho_b )\f$ as
  \f$\alpha\f$ incesases from zero.  The stepsize calculation is
  straightforward, and exploits the fact that \f$\phi ( \alpha )\f$ is a
  piecewise quadratic function of \f$\alpha\f$.

  The search direction is defined by a subset of the "active" terms in
  \f$v(x)\f$, i.e., those for which \f$a_i^T x = c_i^l\f$ or \f$c_i^u\f$
  (for \f$i=1,\ldots ,m\f$) or \f$x_j = x_j^l\f$ or \f$x_j^u\f$ (for
  {j=1,\ldots ,n}).  The "working" set \f$W^{(k)}\f$ is chosen from the
  active terms, and is such that its members have linearly independent
  gradients.  The search direction \f$s^{(k)}\f$ is chosen as an
  approximate solution of the equality-constrained quadratic program
\latexonly
  \[
  (6) \;\; \minin{s \in \smallRe^n} q(x^{(k)} + s) +
   \rho_g l_g^{(k)} (s) + \rho_b l_b^{(k)} (s),
  \]
\endlatexonly
\htmlonly
  $$
  (6) \;\; \min{s \in R^n} q(x^{(k)} + s) +
   \rho_g l_g^{(k)} (s) + \rho_b l_b^{(k)} (s),
  $$
\endhtmlonly
\manonly
  \n
  (6) min_{s in R^n} q(x^{(k)}+s) + rho_g l_g^{(k)}(s)
                                  + rho_b l_b^{(k)}(s),
  \n
\endmanonly
  subject to
\latexonly
  \[
  (7) \;\; a_i^T s = 0,\;\;  i \in \{ 1, \ldots , m \} \cap W^{(k)},
  \;\mbox{and}\;
  x_j = 0, \;\;  i  \in \{1, \ldots , n \} \cap W^{(k)},
  \]
\endlatexonly
\htmlonly
  $$
  (7) \;\; a_i^T s = 0,\;\;  i \in \{ 1, \ldots , m \} \cap W^{(k)},
  \;\mbox{and}\;
  x_j = 0, \;\;  i  \in \{1, \ldots , n \} \cap W^{(k)},
  $$
\endhtmlonly
\manonly
  \n
  (7)  a_i^T s = 0, i in {1,...,m} intersection W^{(k)},
       and x_j = 0, i in {1,...,n} intersection W^{(k)},
  \n
\endmanonly
  where
\latexonly
  \[
  l_g^{(k)} (s) =
       - \sum_{\stackrel{i=1}{a_i^T x < c_i^l}}^m a_i^T s
      \; + \sum_{\stackrel{i=1}{a_i^T x > c_i^u}}^m a_i^T s
  \]
\endlatexonly
\htmlonly
  $$
  l_g^{(k)} (s) =
       - \sum_{\stackrel{i=1}{a_i^T x < c_i^l}}^m a_i^T s
      \; + \sum_{\stackrel{i=1}{a_i^T x > c_i^u}}^m a_i^T s
  $$
\endhtmlonly
\manonly
  \n
  l_g^{(k)}(s) = - sum_{i=1,a_i^T x < c_i^l}^m a_i^T s
                 + sum_{i=1,a_i^T x > c_i^u}^m a_i^T s
  \n
\endmanonly
  and
\latexonly
  \[
  l_b^{(k)} (s) = - \sum_{\stackrel{j=1}{x_j < x_j^l}}^n s_j
      \; + \sum_{\stackrel{j=1}{x_j > x_j^u}}^n s_j.
  \]
\endlatexonly
\htmlonly
  $$
  l_b^{(k)} (s) = - \sum_{\stackrel{j=1}{x_j < x_j^l}}^n s_j
      \; + \sum_{\stackrel{j=1}{x_j > x_j^u}}^n s_j.
  $$
\endhtmlonly
\manonly
  \n
  l_b^{(k)}(s) = - sum_{j=1,x_j < x_j^l}^n s_j
                 + sum_{j=1,x_j > x_j^u}^n s_j.
  \n
\endmanonly
  The equality-constrained quadratic program (6)-(7) is
  solved by a projected preconditioned conjugate gradient method. The
  method terminates either after a prespecified number of iterations, or
  if the solution is found, or if a direction of infinite descent, along which
  \f$q(x^{(k)} + s) + \rho_g l_g^{(k)} (s) + \rho_b l_b^{(k)} (s)\f$
  decreases without bound within the feasible region
  (7), is located.  Succesively more accurate approximations are
  required as suspected solutions of (1) are approached.

  Preconditioning of the conjugate gradient iteration
  requires the solution of one or more linear systems of the form
\latexonly
  \[
  (8)\;\; \mat{cc}{M^{(k)} & A^{(k)T} \\ A^{(k)} & 0 }
  \vect{ p \\ u} = \vect{ g \\ 0 },
  \]
\endlatexonly
\htmlonly
  $$
  (8) \begin{pmatrix} M^{(k)} & A^{(k)T} \\ A^{(k)} & 0 \end{pmatrix}
      \begin{pmatrix} p \\ u \end{pmatrix} =
      \begin{pmatrix} g \\ 0 \end{pmatrix}
  $$
\endhtmlonly
\manonly
  \n
  (8) ( M^{(k)} A^{(k)T} ) ( p ) = ( g )
      ( A^{(k)}    0     ) ( u )   ( 0 )
  \n
\endmanonly
  where \f$M^{(k)}\f$ is a "suitable" approximation to \f$H\f$ and the
  rows of \f$A^{(k)}\f$ comprise the gradients of the terms in the
  current working set. Rather than recomputing a factorization of the
  preconditioner at every iteration, a Schur complement method is used,
  recognising the fact that gradual changes occur to successive working
  sets. The main iteration is divided into a sequence of "major"
  iterations.  At the start of each major iteration (say, the overall
  iteration \f$l\f$), a factorization of the current "reference" matrix,
  that is the matrix
\latexonly
  \[
  (9)\;\; \mat{cc}{M^{(l)} & A^{(l)T} \\ A^{(l)} & 0 }
  \]
\endlatexonly
\htmlonly
  $$
  (9) \begin{pmatrix} M^{(l)} & A^{(l)T} \\ A^{(l)} & 0 \end{pmatrix}
  $$
\endhtmlonly
\manonly
  \n
   (9) ( M^{(l)}   A^{(l)T} )
       ( A^{(l)}      0 }   )
  \n
\endmanonly
  is obtained using the GALAHAD matrix factorization package SLS.
  This reference matrix may be factorized as a whole (the so-called
  "augmented system" approach), or by performing a block elimination first
  (the "Schur-complement" approach). The latter is usually to be preferred
  when a (non-singular) diagonal
  preconditioner is used, but may be inefficient if any of the columns
  of \f$A^{(l)}\f$ is too dense.
  Subsequent iterations within the current major
  iteration obtain solutions to (8) via the factors of (9)
  and an appropriate (dense) Schur complement,
  obtained from the GALAHAD package SCU.
  The major iteration terminates
  once the space required to hold the factors of the (growing) Schur
  complement exceeds a given threshold.

  The working set changes by (a) adding an active term encountered
  during the determination of the stepsize, or (b) the removal of a term
  if \f$s = 0\f$ solves (6)-(7). The decision on which to
  remove in the latter case is based upon the expected decrease upon the
  removal of an individual term, and this information is available from
  the magnitude and sign of the components of the auxiliary vector
  \f$u\f$ computed in (8). At optimality, the components of \f$u\f$
  for \f$a_i\f$ terms will all lie between \f$0\f$ and \f$\rho_g\f$---and those
  for the other terms between \f$0\f$ and \f$\rho_b\f$---and any violation of
  this rule indicates further progress is possible. The components of
  \f$u\f$ corresonding to the terms involving \f$a_i^T x\f$ are
  sometimes known as Lagrange multipliers (or generalized gradients) and
  denoted by \f$y\f$, while those for the remaining \f$x_j\f$ terms are dual
  variables and denoted by \f$z\f$.

  To solve (2)-(4), a sequence of problems of the form
  (1) are solved, each with a larger value of \f$\rho_g\f$ and/or
  \f$\rho_b\f$ than its predecessor. The required solution has been found
  once the infeasibilities \f$v_g(x)\f$ and \f$v_b(x)\f$ have been reduced
  to zero at the solution of (1) for the given \f$\rho_g\f$ and
  \f$\rho_b\f$.

  In order to make the solution as efficient as possible, the variables
  and constraints are reordered internally by the GALAHAD package QPP
  prior to solution.  In particular, fixed variables and free (unbounded
  on both sides) constraints are temporarily removed.


  \subsection qpa_references Reference

  The method is described in detail in

  N. I. M. Gould and Ph. L. Toint (2001).
  ``An iterative working-set method
  for large-scale non-convex quadratic programming''.
  \em Applied Numerical Mathematics
  <b>43 (1-2)</b> (2002) 109--128.

  \subsection qpa_call_order Call order

  To solve a given problem, functions from the qpa package must be called
  in the following order:

  - \link qpa_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link qpa_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link qpa_import \endlink - set up problem data structures and fixed
      values
  - \link qpa_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
    - \link qpa_solve_qp \endlink - solve the quadratic program (2)-(4)
    - \link qpa_solve_l1qp \endlink - solve the l1 quadratic program (1)
    - \link qpa_solve_bcl1qp \endlink - solve the bound constrained
            l1 quadratic program (4)-(5)
  - \link qpa_information \endlink (optional) - recover information about
    the solution and solution process
  - \link qpa_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_QPA_H
#define GALAHAD_QPA_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"

/**
 * control derived type as a C struct
 */
struct qpa_control_type {

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
    /// the factorization to be used. Possible values are
    /// 0  automatic
    /// 1  Schur-complement factorization
    /// 2  augmented-system factorization
    ipc_ factor;

    /// \brief
    /// the maximum number of nonzeros in a column of A which is permitted
    /// with the Schur-complement factorization
    ipc_ max_col;

    /// \brief
    /// the maximum permitted size of the Schur complement before a
    /// refactorization is performed
    ipc_ max_sc;

    /// \brief
    /// an initial guess as to the integer workspace required by SLS  (OBSOLETE)
    ipc_ indmin;

    /// \brief
    /// an initial guess as to the real workspace required by SLS     (OBSOLETE)
    ipc_ valmin;

    /// \brief
    /// the maximum number of iterative refinements allowed           (OBSOLETE)
    ipc_ itref_max;

    /// \brief
    /// the infeasibility will be checked for improvement every
    /// infeas_check_interval iterations (see infeas_g_improved_by_factor
    /// and infeas_b_improved_by_factor below)
    ipc_ infeas_check_interval;

    /// \brief
    /// the maximum number of CG iterations allowed. If cg_maxit < 0,
    /// this number will be reset to the dimension of the system + 1
    ipc_ cg_maxit;

    /// \brief
    /// the preconditioner to be used for the CG is defined by precon.
    /// Possible values are
    /// 0  automatic
    /// 1  no preconditioner, i.e, the identity within full factorization
    /// 2  full factorization
    /// 3  band within full factorization
    /// 4  diagonal using the barrier terms within full factorization
    ipc_ precon;

    /// \brief
    /// the semi-bandwidth of a band preconditioner, if appropriate
    ipc_ nsemib;

    /// \brief
    /// if the ratio of the number of nonzeros in the factors of the reference
    /// matrix to the number of nonzeros in the matrix itself exceeds
    /// full_max_fill, and the preconditioner is being selected automatically
    /// (precon = 0), a banded approximation will be used instead
    ipc_ full_max_fill;

    /// \brief
    /// the constraint deletion strategy to be used. Possible values are:
    ///
    /// 0  most violated of all
    /// 1  LIFO (last in, first out)
    /// k  LIFO(k) most violated of the last k in LIFO
    ipc_ deletion_strategy;

    /// \brief
    /// indicate whether and how much of the input problem should be restored
    /// on output. Possible values are
    /// 0 nothing restored
    /// 1 scalar and vector parameters
    /// 2 all parameters
    ipc_ restore_problem;

    /// \brief
    /// the frequency at which residuals will be monitored
    ipc_ monitor_residuals;

    /// \brief
    ///
    /// indicates whether a cold or warm start should be made.
    /// Possible values are
    ///
    /// 0 warm start - the values set in C_stat and B_stat indicate which
    /// constraints will be included in the initial working set.
    /// 1 cold start from the value set in X; constraints active
    /// at X will determine the initial working set.
    /// 2 cold start with no active constraints
    /// 3 cold start with only equality constraints active
    /// 4 cold start with as many active constraints as possible
    ipc_ cold_start;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    ipc_ sif_file_device;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// any constraint violated by less than feas_tol will be considered to be
    /// satisfied
    rpc_ feas_tol;

    /// \brief
    /// if the objective function value is smaller than obj_unbounded, it will
    /// be flagged as unbounded from below.
    rpc_ obj_unbounded;

    /// \brief
    /// if the problem is currently infeasible and solve_qp (see below) is
    /// .TRUE. the current penalty parameter for the general constraints will
    /// be increased by increase_rho_g_factor when needed
    rpc_ increase_rho_g_factor;

    /// \brief
    /// if the infeasibility of the general constraints has not dropped by a fac
    /// of infeas_g_improved_by_factor over the previous infeas_check_interval
    /// iterations, the current corresponding penalty parameter will be increase
    rpc_ infeas_g_improved_by_factor;

    /// \brief
    /// if the problem is currently infeasible and solve_qp or solve_within_boun
    /// (see below) is .TRUE., the current penalty parameter for the simple boun
    /// constraints will be increased by increase_rho_b_factor when needed
    rpc_ increase_rho_b_factor;

    /// \brief
    /// if the infeasibility of the simple bounds has not dropped by a factor of
    /// infeas_b_improved_by_factor over the previous infeas_check_interval
    /// iterations, the current corresponding penalty parameter will be increase
    ///
    rpc_ infeas_b_improved_by_factor;

    /// \brief
    /// the threshold pivot used by the matrix factorization.
    /// See the documentation for SLS for details                        (OBSOLE
    rpc_ pivot_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization when attempting to
    /// detect linearly dependent constraints.
    rpc_ pivot_tol_for_dependencies;

    /// \brief
    /// any pivots smaller than zero_pivot in absolute value will be regarded to
    /// zero when attempting to detect linearly dependent constraints    (OBSOLE
    rpc_ zero_pivot;

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
    /// any dual variable or Lagrange multiplier which is less than multiplier_t
    /// outside its optimal interval will be regarded as being acceptable when
    /// checking for optimality
    rpc_ multiplier_tol;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// any problem bound with the value zero will be treated as if it were a
    /// general value if true
    bool treat_zero_bounds_as_general;

    /// \brief
    /// if solve_qp is .TRUE., the value of prob.rho_g and prob.rho_b will be
    /// increased as many times as are needed to ensure that the output
    /// solution is feasible, and thus aims to solve the quadratic program
    /// (2)-(4)
    bool solve_qp;

    /// \brief
    /// if solve_within_bounds is  .TRUE., the value of prob.rho_b will be
    /// increased as many times as are needed to ensure that the output
    /// solution is feasible with respect to the simple bounds, and thus
    /// aims to solve the bound-constrained quadratic program (4)-(5)
    bool solve_within_bounds;

    /// \brief
    /// if randomize is .TRUE., the constraint bounds will be perturbed by
    /// small random quantities during the first stage of the solution
    /// process. Any randomization will ultimately be removed. Randomization
    /// helps when solving degenerate problems
    bool randomize;

    /// \brief
    /// if .array_syntax_worse_than_do_loop is true, f77-style do loops will be
    /// used rather than f90-style array syntax for vector operations (OBSOLETE)
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
    /// indefinite linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// definite linear equation solver
    //char definite_linear_solver[31];

    // \brief
    /// name of generated SIF file containing input problem
    char sif_file_name[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// component specifically for parametric problems (not used at present)
    bool each_interval;

    /// \brief
    /// control parameters for SLS
    struct sls_control_type sls_control;
};

/**
 * time derived type as a C struct
 */
struct qpa_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    rpc_ preprocess;

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
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    rpc_ clock_preprocess;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
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
struct qpa_inform_type {

    /// \brief
    /// return status. See QPA_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of major iterations required
    ipc_ major_iter;

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
    /// the total number of factorizations which were modified to ensure that th
    /// matrix was an appropriate preconditioner
    ipc_ nmods;

    /// \brief
    /// the number of infeasible general constraints
    ipc_ num_g_infeas;

    /// \brief
    /// the number of infeasible simple-bound constraints
    ipc_ num_b_infeas;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by QPA_solve
    rpc_ obj;

    /// \brief
    /// the 1-norm of the infeasibility of the general constraints
    rpc_ infeas_g;

    /// \brief
    /// the 1-norm of the infeasibility of the simple-bound constraints
    rpc_ infeas_b;

    /// \brief
    /// the merit function value = obj + rho_g * infeas_g + rho_b * infeas_b
    rpc_ merit;

    /// \brief
    /// timings (see above)
    struct qpa_time_type time;

    /// \brief
    /// inform parameters for SLS
    struct sls_inform_type sls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    Q P A  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void qpa_initialize( void **data,
                     struct qpa_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see qpa_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    Q P A  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void qpa_read_specfile( struct qpa_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNQPA.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/qpa.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see qpa_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    Q P A  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void qpa_import( struct qpa_control_type *control,
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
  paramters for the remaining prcedures (see qpa_control_type)

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


//  *-*-*-*-*-*-*-*-*-   Q P A _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void qpa_reset_control( struct qpa_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see qpa_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  *-*-*-*-*-*-*-*-*-*-*-   Q P A _ S O L V E _ Q P   -*-*-*-*-*-*-*-*-*-*-*-*

void qpa_solve_qp( void **data,
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
 Solve the quadratic program (2)-(4).

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

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the current status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds. On entry, if control.cold_start = 0, x_stat should
    be set as above to provide a guide to the initial working set.

 @param[in,out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the current status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^Tx\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies between its bounds. On entry, if control.cold_start = 0,
    c_stat should be set as above to provide a guide to the initial working set.
*/

//  *-*-*-*-*-*-*-*-*-*-   Q P A _ S O L V E _ L 1 Q P   -*-*-*-*-*-*-*-*-*-*-

void qpa_solve_l1qp( void **data,
                     ipc_ *status,
                     ipc_ n,
                     ipc_ m,
                     ipc_ h_ne,
                     const rpc_ H_val[],
                     const rpc_ g[],
                     const rpc_ f,
                     const rpc_ rho_g,
                     const rpc_ rho_b,
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
 Solve the l_1 quadratic program (1).

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

 @param[in] rho_g is a scalar of type rpc_, that
    holds the parameter \f$\rho_g\f$ associated with the linear constraints.

 @param[in] rho_b is a scalar of type rpc_, that holds the parameter
   \f$\rho_b\f$ associated with the simple bound constraints.

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

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the current status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds. On entry, if control.cold_start = 0, x_stat should
    be set as above to provide a guide to the initial working set.

 @param[in,out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the current status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^Tx\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies between its bounds. On entry, if control.cold_start = 0,
    c_stat should be set as above to provide a guide to the initial working set.
*/

//  *-*-*-*-*-*-*-*-*-*-   Q P A _ S O L V E _ B C L 1 Q P   -*-*-*-*-*-*-*-*-

void qpa_solve_bcl1qp( void **data,
                       ipc_ *status,
                       ipc_ n,
                       ipc_ m,
                       ipc_ h_ne,
                       const rpc_ H_val[],
                       const rpc_ g[],
                       const rpc_ f,
                       const rpc_ rho_g,
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
 Solve the bound-constrained l_1 quadratic program (4)-(5)

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

 @param[in] rho_g is a scalar of type rpc_, that
    holds the parameter \f$\rho_g\f$ associated with the linear constraints.

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

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the current status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds. On entry, if control.cold_start = 0, x_stat should
    be set as above to provide a guide to the initial working set.

 @param[in,out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the current status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^Tx\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies between its bounds. On entry, if control.cold_start = 0,
    c_stat should be set as above to provide a guide to the initial working set.
*/

// *-*-*-*-*-*-*-*-*-*-    Q P A  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void qpa_information( void **data,
                      struct qpa_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see qpa_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    Q P A  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void qpa_terminate( void **data,
                    struct qpa_control_type *control,
                    struct qpa_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see qpa_control_type)

  @param[out] inform   is a struct containing output information
              (see qpa_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example qpat.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example qpatf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
