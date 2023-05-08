QPA
===

.. module:: galahad.qpa

The ``qpa`` package uses a **working-set method** to solve
**non-convex quadratic programs** in various guises.
The first is the **$\mathbf{\ell_1}$ quadratic programming problem** 
that aims to minimize
$$f(x;\rho_g,\rho_b) = q(x) + \rho_g v_g(x) + \rho_b v_b(x)$$
involving the quadratic objective
$$q(x) = f + g^T x + \frac{1}{2} x^T H x$$
and the infeasibilities
$$v_g(x) = \sum_{i=1}^{m} \max(c_i^l - a_i^T x, 0) 
+ \sum_{i=1}^{m} \max(a_i^T x - c_i^u, 0)$$
and
$$v_b(x) = \sum_{j=1}^{n} \max(x_j^l - x_j  , 0) 
+ \sum_{j=1}^{n} \max(x_j  - x_j^u , 0),$$
where the $n$ by $n$ symmetric matrix $H$, the 
vectors $g$, $a_i$, $c^l$, $c^u$, $x^l$, $x^u$ 
and the scalars $f$, $\rho_g$ and $\rho_b$ are given.
Full advantage is taken of any zero coefficients in the matrices $H$ 
or $A$ (whose rows are the vectors $a_i^T$).
Any of the constraint bounds $c_i^l$, $c_i^u$, 
$x_j^l$ and $x_j^u$ may be infinite.

The package may also be used to solve the 
**standard quadratic programming problem**
whose aim is to minimize $q(x)$
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
by automatically adjusting the parameters $\rho_g$ and $\rho_b$ in
$f(x;\rho_g,\rho_b)$.

Similarly, the package is capable of solving the 
**bound-constrained $\mathbf{\ell_1}$ quadratic programming problem**
whose intention is to minimize $q(x) + \rho_g v_g(x),$
subject to the above simple bound constraints
by automatically adjusting $\rho_b$ in $f(x;\rho_g,\rho_b)$.

If the matrix $H$ is positive semi-definite, a global solution is found. 
However, if $H$ is indefinite, the procedure may find a (weak second-order) 
critical point that is not the global solution to the given problem.

See Section 4 of $GALAHAD/doc/qpa.pdf for additional details.

**N.B.** In many cases, the alternative quadratic programming package ``qpb`` 
is faster, and thus to be preferred.

terminolgy
----------

Any required solution $x$ for the standard quadratic programming problem
necessarily satisfies the **primal optimality conditions**
$$A x = c$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$H x + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; z = z_l + z_u,$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

At the $k$-th iteration of the method, an improvement to the value
of the merit function 
$m(x, \rho_g, \rho_b ) = q(x) + \rho_g v_g(x) + \rho_b v_b(x)$
at $x = x^{(k)}$ is sought. This is achieved by first 
computing a search direction $s^{(k)}$,
and then setting $x^{(k+1)} = x^{(k)} + \alpha^{(k)} s^{(k)}$,
where the stepsize $\alpha^{(k)}$ is chosen as the first local minimizer of 
$\phi ( \alpha ) = m( x^{(k)} + \alpha s^{(k)} , \rho_g, \rho_b )$
as $\alpha$ incesases from zero. 
The stepsize calculation is straightforward, and exploits the fact that
$\phi ( \alpha )$ is a piecewise quadratic function of $\alpha$.

The search direction is defined by a subset of the "active" terms in 
$v(x)$, i.e., those for which 
$a_i^T x = c_i^l$ or $c_i^u$ (for $i=1,\ldots,m$) or 
$x_j = x_j^l$ or $x_j^u$ (for ($j=1,\ldots,n$).
The "working" set $W^{(k)}$ is chosen from the active terms, and is such 
that its members have linearly independent gradients. 
The search direction $s^{(k)}$ is chosen as an approximate solution of 
the equality-constrained quadratic program
$$
{\renewcommand{\arraystretch}{0.8}
\begin{array}[t]{c}
\mbox{minimize} \\
\mbox{ $s \in R^n $ }
\end{array} \;}
q(x^{(k)} + s) + 
\rho_g l_g^{(k)} (s) + \rho_b l_b^{(k)} (s),\;\;\mbox{(1)}$$
subject to 
$$a_i^T s = 0,\;\;  i \in \{ 1, \ldots , m \} \cap W^{(k)},
\;\;\mbox{and}\;\;
x_j = 0, \;\;  i  \in \{1, \ldots , n \} \cap W^{(k)},\;\;\mbox{(2)}$$
where
$$l_g^{(k)} (s) = - \sum_{\stackrel{i=1}{a_i^T x < c_i^l}}^m a_i^T s 
\; + \sum_{\stackrel{i=1}{a_i^T x > c_i^u}}^m a_i^T s$$
and
$$l_b^{(k)} (s) = - \sum_{\stackrel{j=1}{x_j < x_j^l}}^n s_j
\; + \sum_{\stackrel{j=1}{x_j > x_j^u}}^n s_j. $$
The equality-constrained quadratic program (1)--(2) is solved by
a projected preconditioned conjugate gradient method. The method terminates
either after a prespecified number of iterations, or if the solution is found,
or if a direction of infinite descent, along which 
$q(x^{(k)} + s) + \rho_g l_g^{(k)} (s) + \rho_b l_b^{(k)} (s)$
decreases without bound within the feasible region (2), is located. 
Succesively more accurate approximations are required as suspected 
solutions of the $\ell_1$-QP are approached.

Preconditioning of the conjugate gradient iteration
requires the solution of one or more linear systems of the form
$$\left(\begin{array}{cc}
M^{(k)} & A^{(k)T} \\ A^{(k)} & 0
\end{array}\right) \left(\begin{array}{c}
p \\ u
\end{array}\right) = \left(\begin{array}{c}
g \\ 0
\end{array}\right),\;\;\mbox{(3)}$$
where $M^{(k)}$ is a "suitable" approximation to $H$ and the rows of
$A^{(k)}$ comprise the gradients of the terms in the current working
set. Rather than recomputing a factorization of the preconditioner at
every iteration, a Schur complement method is used, recognising the fact
that gradual changes occur to successive working sets. The main
iteration is divided into a sequence of "major" iterations.  At the
start of each major iteration (say, at the overall iteration $l$), a
factorization of the current "reference" matrix, that is the matrix
$$K^{(l)} = \left(\begin{array}{cc}
M^{(l)} & A^{(l)T} \\ A^{(l)} & 0 
\end{array}\right)$$
is obtained using the matrix factorization package ``SLS``.  This
reference matrix may be factorized as a whole (the so-called "augmented
system" approach), or by performing a block elimination first (the
"Schur-complement" approach). The latter is usually to be preferred when
a (non-singular) diagonal preconditioner is used, but may be inefficient
if any of the columns of $A^{(l)}$ is too dense.  Subsequent iterations
within the current major iteration obtain solutions to (3) via the
factors of $K^{(l)}$ and an appropriate (dense) Schur complement,
obtained from the package ``SCU``.  The major iteration terminates once
the space required to hold the factors of the (growing) Schur complement
exceeds a given threshold.

The working set changes by (a) adding an active term encountered during 
the determination of the stepsize, or (b) the removal of a term if $s = 0$
solves (1)--(2). The  decision on which to remove in the latter 
case is based upon the expected decrease upon the removal of an individual term,
and this information is available from the magnitude and sign of the components
of the auxiliary vector $u$ computed in (3). At optimality, the
components of $u$ for $a_i$ terms will all lie between 
$0$ and $\rho_g$ --- and those for the other terms 
between $0$ and $\rho_b$ --- and any violation
of this rule indicates further progress is possible. The components
of $u$ corresonding to the terms involving $a_i^T x$
are sometimes known as Lagrange multipliers (or generalized gradients) and
denoted by $y$, while those for the remaining $x_j$ terms are dual variables
and denoted by $z$.

To solve the standard quadratic programming problem, a sequence of 
$\ell_1$-quadratic programs are solved, each with a larger value of 
$\rho_g$ and/or $\rho_b$ than its predecessor. The
required solution has been found once the infeasibilities 
$v_g(x)$ and $v_b(x)$ have been reduced to zero at the solution of 
the $\ell_1$-problem for the given $\rho_g$ and $\rho_b$.

In order to make the solution as efficient as possible, the variables
and constraints are reordered internally by the package ``QPP`` prior
to solution. In particular, fixed variables and free (unbounded on 
both sides) constraints are temporarily removed.

reference
---------

The method is described in detail in

  N. I. M. Gould and Ph. L. Toint
  ``An iterative working-set method for large-scale 
  non-convex quadratic programming''.
  *Applied Numerical Mathematics* **43(1--2)** (2002) 109--128.


matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ may be presented
and stored in a variety of convenient input formats. 

*Dense* storage format:
The matrix $A$ is stored as a compact dense matrix by rows, that is,
the values of the entries of each row in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $n \ast i + j$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense' should be specified.

*Dense by columns* storage format:
The matrix $A$ is stored as a compact dense matrix by columns, that is,
the values of the entries of each column in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $m \ast j + i$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense_by_columns' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as the $l$-th 
components of the integer arrays A_row and A_col and real array A_val, 
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
The string A_type = 'coordinate'should be specified.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.
For sparse matrices, this scheme almost always requires less storage than
its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Sparse column-wise* storage format:
Once again only the nonzero entries are stored, but this time
they are ordered so that those in column j appear directly before those
in column j+1. For the j-th column of $A$ the j-th component of the
integer array A_ptr holds the position of the first entry in this column,
while A_ptr(n) holds the total number of entries.
The row indices i, $0 \leq i \leq m-1$, and values $A_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = A_ptr(j), $\ldots$, A_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array A_row, and real array A_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.
The string A_type = 'sparse_by_columns' should be specified.

The **symmetric** $n$ by $n$ matrix $H$ may also
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string H_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $H$,
its row index i, column index j and value $H_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string H_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string H_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $H$ is diagonal (i.e., $H_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$H_{ii}$, $0 \leq i \leq n-1$ need be stored, 
and the first n components of the array H_val may be used for the purpose.
The string H_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $H$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of H_val.
The string H_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $H$ is the identity matrix, no values need be stored.
The string H_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $H$ is the zero matrix, but now
the string H_type = 'zero' or 'none' should be specified.


functions
---------

   .. function:: qpa.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output,

             * **1**

               gives a one-line summary for every iteration.

             * **2**

               gives a summary of the inner iteration for each iteration.

             * **>=3**

               gives increasingly verbose (debugging) output.

          start_print : int
             any printing will start on this iteration.
          stop_print : int
             any printing will stop on this iteration.
          maxit : int
             at most maxit inner iterations are allowed.
          factor : int
             the factorization to be used. Possible values are 0
             automatic 1 Schur-complement factorization 2
             augmented-system factorization.
          max_col : int
             the maximum number of nonzeros in a column of A which is
             permitted with the Schur-complement factorization.
          max_sc : int
             the maximum permitted size of the Schur complement before
             a refactorization is performed.
          indmin : int
             an initial guess as to the integer workspace required by
             SLS (OBSOLETE).
          valmin : int
             an initial guess as to the real workspace required by SLS
             (OBSOLETE).
          itref_max : int
             the maximum number of iterative refinements allowed
             (OBSOLETE).
          infeas_check_interval : int
             the infeasibility will be checked for improvement every
             infeas_check_interval iterations (see
             infeas_g_improved_by_factor and
             infeas_b_improved_by_factor below).
          cg_maxit : int
             the maximum number of CG iterations allowed. If cg_maxit <
             0, this number will be reset to the dimension of the
             system + 1.
          precon : int
             the preconditioner to be used for the CG is defined by
             precon. Possible values are 0 automatic 

             * **1**

               no preconditioner, i.e, the identity within full factorization 

             * **2**

               full factorization 

             * **3**

               band within full factorization 

             * **4**

               diagonal using the barrier terms within full factorization.

          nsemib : int
             the semi-bandwidth of a band preconditioner, if
             appropriate.
          full_max_fill : int
             if the ratio of the number of nonzeros in the factors of
             the reference matrix to the number of nonzeros in the
             matrix itself exceeds full_max_fill, and the
             preconditioner is being selected automatically (precon =
             0), a banded approximation will be used instead.
          deletion_strategy : int
             the constraint deletion strategy to be used. Possible
             values are:  0 most violated of all 1 LIFO (last in, first
             out) k LIFO(k) most violated of the last k in LIFO.
          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are 0 nothing
             restored 1 scalar and vector parameters 2 all parameters.
          monitor_residuals : int
             the frequency at which residuals will be monitored.
          cold_start : int
             indicates whether a cold or warm start should be made.
             Possible values are

             * **0**

               warm start - the values set in C_stat and B_stat indicate 
               which constraints will be included in the initial working set. 

             * **1**

               cold start from the value set in X; constraints active at X 
               will determine the initial working set. 

             * **2**

               cold start with no active constraints 

             * **3**

               cold start with only equality constraints active 

             * **4**

               cold start with as many active constraints as possible.

          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          feas_tol : float
             any constraint violated by less than feas_tol will be
             considered to be satisfied.
          obj_unbounded : float
             if the objective function value is smaller than
             obj_unbounded, it will b flagged as unbounded from below.
          increase_rho_g_factor : float
             if the problem is currently infeasible and solve_qp (see
             below) is True, the current penalty parameter for the
             general constraints will be increased by
             increase_rho_g_factor when needed.
          infeas_g_improved_by_factor : float
             if the infeasibility of the general constraints has not
             dropped by a fac of infeas_g_improved_by_factor over the
             previous infeas_check_interval iterations, the current
             corresponding penalty parameter will be increase.
          increase_rho_b_factor : float
             if the problem is currently infeasible and solve_qp or
             solve_within_boun (see below) is True,` the current
             penalty parameter for the simple boun constraints will be
             increased by increase_rho_b_factor when needed.
          infeas_b_improved_by_factor : float
             if the infeasibility of the simple bounds has not dropped
             by a factor of infeas_b_improved_by_factor over the
             previous infeas_check_interval iterations, the current
             corresponding penalty parameter will be increase.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SLS for details (OBSOLETE).
          pivot_tol_for_dependencies : float
             the threshold pivot used by the matrix factorization when
             attempting to detect linearly dependent constraints.
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints (OBSOLETE).
          inner_stop_relative : float
             the search direction is considered as an acceptable
             approximation to the minimizer of the model if the
             gradient of the model in the preconditioning(inverse) norm
             is less than max( inner_stop_relative * initial
             preconditioning(inverse) gradient norm,
             inner_stop_absolute ).
          inner_stop_absolute : float
             see inner_stop_relative.
          multiplier_tol : float
             any dual variable or Lagrange multiplier which is less
             than multiplier_t outside its optimal interval will be
             regarded as being acceptable when checking for optimality.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          treat_zero_bounds_as_general : bool
             any problem bound with the value zero will be treated as
             if it were a general value if True.
          solve_qp : bool
             if solve_qp is True, the value of prob.rho_g and
             prob.rho_b will be increased as many times as are needed
             to ensure that the output solution is feasible, and thus
             aims to solve the quadratic program (2)-(4).
          solve_within_bounds : bool
             if solve_within_bounds is True, the value of
             prob.rho_b will be increased as many times as are needed
             to ensure that the output solution is feasible with
             respect to the simple bounds, and thus aims to solve the
             bound-constrained quadratic program (4)-(5).
          randomize : bool
             if randomize is True, the constraint bounds will be
             perturbed by small random quantities during the first
             stage of the solution process. Any randomization will
             ultimately be removed. Randomization helps when solving
             degenerate problems.
          array_syntax_worse_than_do_loop : bool
             if ``array_syntax_worse_than_do_loop`` is True, f77-style
             do loops will be used rather than f90-style array syntax
             for vector operations (OBSOLETE).
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if ``generate_sif_file`` is True, a SIF file
             describing the current problem is to be generated.
          symmetric_linear_solver : str
             indefinite linear equation solver.
          definite_linear_solver : str
             definite linear equation solver.
          sif_file_name : str
             name of generated SIF file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          each_interval : bool
             component specifically for parametric problems (not used
             at present).
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).

   .. function:: qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
      H_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      H_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      H_row : ndarray(H_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      H_col : ndarray(H_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      A_type : string
          specifies the unsymmetric storage scheme used for the constraints 
          Jacobian $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in $A$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other two schemes.
      A_row : ndarray(A_ne)
          holds the row indices of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other two schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of $A$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when the 
          dense storage scheme is used, and in this case can be None.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries, in the sparse row-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      options : dict, optional
          dictionary of control options (see ``qpa.initialize``).

   .. function:: qpa.solve_qp(n, m, f, g, h_ne, H_val, a_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the standard non-convex quadratic program 
      involving the quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      h_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(h_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: qpa.solve_l1qp(n, m, f, g, h_ne, H_val, rho_g, rho_b, a_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the non-convex quadratic program involving the
      $\mathbf{\ell_1}$ quadratic objective function $f(x;\rho_g,\rho_b)$
      for given $\rho_g$. and $\rho_b$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      h_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(h_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      rho_g : float
          holds the weight $\rho_g$ associated with the general 
          infeasibilities
      rho_b : float
          holds the weight $\rho_b$ associated with the simple bound
          infeasibilities
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: qpa.solve_bcl1qp(n, m, f, g, h_ne, H_val, rho_g, a_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a local solution of the non-convex quadratic program involving the
      bound-constrained $\mathbf{\ell_1}$ quadratic objective function 
      $q(x) + \rho_g v_g(x),$ for given $\rho_g$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      h_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(h_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      rho_g : float
          holds the weight $\rho_g$ associated with the general 
          infeasibilities
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``qpa.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] qpa.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            return status.  Possible values are:

            * **0**

              The run was succesful.

            * **-1**

              An allocation error occurred. A message indicating the
              offending array is written on unit control['error'], and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-2**

              A deallocation error occurred.  A message indicating the
              offending array is written on unit control['error'] and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-3**

              The restriction n > 0 or m > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
              has been violated.

            * **-4**

              The bound constraints are inconsistent.

            * **-5**

              The constraints appear to have no feasible point.

            * **-7**

              The objective function appears to be unbounded from below
              on the feasible set.

            * **-9**

              The analysis phase of the factorization failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              The factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform['factor_status'].

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-17**

              The step is too small to make further progress.

            * **-18**

              Too many iterations have been performed. This may happen if
              control['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              control['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          major_iter : int
             the total number of major iterations required.
          iter : int
             the total number of iterations required.
          cg_iter : int
             the total number of conjugate gradient iterations required.
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          nmods : int
             the total number of factorizations which were modified to
             ensure that th matrix was an appropriate preconditioner.
          num_g_infeas : int
             the number of infeasible general constraints.
          num_b_infeas : int
             the number of infeasible simple-bound constraints.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by QPA_solve.
          infeas_g : float
             the 1-norm of the infeasibility of the general constraints.
          infeas_b : float
             the 1-norm of the infeasibility of the simple-bound
             constraints.
          merit : float
             the merit function value = obj + rho_g * infeas_g + rho_b
             * infeas_b.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorizatio.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorizat.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).


   .. function:: qpa.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/qpa/Python/test_qpa.py
   :code: python

This example code is available in $GALAHAD/src/qpa/Python/test_qpa.py .
