purpose
-------

The ``qpa`` package uses a **working-set method** to solve
**non-convex quadratic programs** in various guises.
The first is the **l${}_1$ quadratic programming problem** 
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
**bound-constrained l${}_1$ quadratic programming problem**
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
