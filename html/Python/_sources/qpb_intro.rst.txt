purpose
-------

The ``qpb`` package uses a 
**primal-dual interior-point method** to solve a given
**non-convex quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric and $m$ by $n$ rectangular matrices,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

If the matrix $H$ is positive semi-definite, a global solution is found. 
However, if $H$ is indefinite, the procedure may find a (weak second-order) 
critical point that is not the global solution to the given problem.

See Section 4 of $GALAHAD/doc/qpb.pdf for additional details.

terminolgy
----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c\;\;\mbox{(1)}$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(2)}$$
the **dual optimality conditions**
$$H x + g = A^{T} y + z,\;\;  y = y_l + y_u 
\;\;\mbox{and}\;\; z = z_l + z_u,\;\;\mbox{(3)}$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 
\;\;\mbox{and}\;\; z_u \leq 0,\;\;\mbox{(4)}$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,\;\;\mbox{(5)}$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

Primal-dual interior point methods iterate towards a point
that satisfies these conditions by ultimately aiming to satisfy
(1), (3) and (5), while ensuring that (2) and (4) are 
satisfied as strict inequalities at each stage. 
Appropriate norms of the amounts by 
which (1), (3) and (5) fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. The fact that (2) and (4) are satisfied as strict 
inequalities gives such methods their other title, namely 
interior-point methods.

The problem is solved in two phases. The goal of the first 
"initial feasible point" phase is
to find a strictly interior point which is primal feasible, that is that
(1) is satisfied. The package ``LSQP`` is used for this 
purpose, and offers the options of either accepting the first 
strictly feasible point found, or preferably of aiming for the
so-called "analytic center" of the feasible region.
Having found such a suitable initial feasible point, the second "optimality"
phase ensures that (1) remains satisfied while iterating to
satisfy dual feasibility (3) and complementary slackness (5).
The optimality phase proceeds by approximately minimizing a 
sequence of barrier functions
$$\frac{1}{2} x^T H x + g^T x + f -
\mu \left[ \sum_{i=1}^{m} \log ( c_{i}  -  c_{i}^{l} )
+ \sum_{i=1}^{m} \log ( c_{i}^{u}  -  c_{i} )
+ \sum_{j=1}^{n} \log ( x_{j}  -  x_{j}^{l} ) 
+ \sum_{j=1}^{n} \log ( x_{j}^{u}  -  x_{j} ) \right] ,$$
for an approriate sequence of positive barrier parameters $\mu$ 
converging to zero while ensuring that (1) remain satisfied and that 
$x$ and $c$ are strictly interior points for (2). 
Note that terms in the above sumations corresponding to infinite bounds 
are ignored, and that equality constraints are treated specially.

Each of the barrier subproblems is solved using a trust-region method.
Such a method generates a trial correction step $\Delta (x, c)$ to the
current iterate $(x, c)$ by replacing the nonlinear barrier function
locally by a suitable quadratic model, and approximately minimizing this
model in the intersection of (1) and a trust region $\|\Delta (x, c)\|
\leq \Delta$ for some appropriate strictly positive trust-region radius
$\Delta$ and norm $\| \cdot \|$.  The step is accepted/rejected and the
radius adjusted on the basis of how accurately the model reproduces the
value of barrier function at the trial step. If the step proves to be
unacceptable, a linesearch is performed along the step to obtain an
acceptable new iterate. In practice, the natural primal "Newton" model
of the barrier function is frequently less successful than an
alternative primal-dual model, and consequently the primal-dual model is
usually to be preferred.

Once a barrier subproblem has been solved, extrapolation based on values
and derivatives encountered on the central path is optionally used to
determine a good starting point for the next subproblem.  Traditional
Taylor-series extrapolation has been superceded by more accurate
Puiseux-series methods as these are particularly suited to deal with
degeneracy.

The trust-region subproblem is approximately solved using the combined
conjugate-gradient/Lanczos method implemented in the package ``GLTR``.
Such a method requires a suitable preconditioner, and in our case, the
only flexibility we have is in approximating the model of the
Hessian. Although using a fixed form of preconditioning is sometimes
effective, we have provided the option of an automatic choice, that aims
to balance the cost of applying the preconditioner against the needs for
an accurate solution of the trust-region subproblem.  The preconditioner
is applied using the matrix factorization package ``SBLS``, but options
at this stage are to factorize the preconditioner as a whole (the
so-called "augmented system" approach), or to perform a block
elimination first (the "Schur-complement" approach). The latter is
usually to be prefered when a (non-singular) diagonal preconditioner is
used, but may be inefficient if any of the columns of $A$ is too dense.

In order to make the solution as efficient as possible, the 
variables and constraints are reordered internally
by the package ``QPP`` prior to solution. 
In particular, fixed variables, and 
free (unbounded on both sides) constraints are temporarily removed.

reference
---------

The method is described in detail in

  A. R. Conn, N. I. M. Gould, D. Orban and Ph. L. Toint,
  ``A primal-dual trust-region algorithm for minimizing a non-convex 
  function subject to general inequality and linear equality constraints''.
  *Mathematical Programming* **87**  (1999) 215-249.
