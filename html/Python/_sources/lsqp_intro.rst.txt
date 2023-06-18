purpose
-------

The ``lsqp`` package uses an **interior-point trust-region method** to solve a
given **linear or separable convex quadratic program**.
The aim is to minimize the separable quadratic objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $A$ is a given $m$ by $n$ matrix,  
$g$, $w$ and $x^0$ are vectors, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

In the special case where $w = 0$, $g = 0$ and $f = 0$,
the so-called *analytic center* of the feasible set will be found,
while *linear programming*, or *constrained least distance*, problems
may be solved by picking $w = 0$, or $g = 0$ and $f = 0$, respectively.

See Section 4 of $GALAHAD/doc/lsqp.pdf for additiional details.

The more-modern package ``cqp`` offers similar functionality, and
is often to be preferred.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c\;\;\mbox{(1a)}$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(1b)}$$
the **dual optimality conditions**
$$W^2 ( x - x^0 ) + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; 
z = z_l + z_u,\;\;\mbox{(2a)}$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; 
z_u \leq 0,\;\;\mbox{(2b)}$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,\;\;\mbox{(3)}$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

Primal-dual interior point methods iterate towards a point that satisfies 
these optimality conditions by ultimately aiming to satisfy
(1a), (2a) and (3), while ensuring that (1b) and (2b) are
satisfied as strict inequalities at each stage.
Appropriate norms of the amounts by
which (1a), (2a) and (3) fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. The fact that (1b) and (2b) are satisfied as strict
inequalities gives such methods their other title, namely
interior-point methods.

When $w \neq 0$ or $g \neq 0$, the method aims at each stage to reduce the
overall violation of (1a), (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; c, \; y, \; y^{l}, \; y^{u}, \; z, \; z^{l}, \; z^{u})$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \; c, \; y, \; y^{l}, \; 
y^{u} ,\;z,\;z^{l} ,\;z^{u} )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (1a), (2a) and a parameterized perturbation
of (3). An improved estimate $v + \alpha \Delta v$
is then used, where the step-size $\alpha$
is chosen as close to 1.0 as possible while ensuring both that
(1b) and (2b) continue to hold and that the individual components
which make up the complementary slackness
(3) do not deviate too significantly
from their average value. The parameter that controls the perturbation
of (3) is ultimately driven to zero.

The Newton equations are solved  by applying the matrix factorization 
package ``SBLS``, but there are options
to factorize the matrix as a whole (the so-called "augmented system"
approach), to perform a block elimination first (the "Schur-complement"
approach), or to let the method itself decide which of the two
previous options is more appropriate.
The "Schur-complement" approach is usually to be preferred when all the
weights are nonzero or when every variable is bounded (at least one side),
but may be inefficient if any of the columns of $A$ is too dense.

When $w = 0$ and $g = 0$, the method aims instead firstly to find an 
interior primal feasible point, that is to ensure that (1a) is satisfied. 
One this has been achieved, attention is switched to mninizing the
potential function
$$\phi (x,\;c) =
- \sum_{i=1}^{m} \log ( c_{i}  -  c_{i}^{l} )
- \sum_{i=1}^{m} \log ( c_{i}^{u}  -  c_{i} )
- \sum_{j=1}^{n} \log ( x_{j}  -  x_{j}^{l} ) 
- \sum_{j=1}^{n} \log ( x_{j}^{u}  -  x_{j} ) ,$$
while ensuring that (1a) remain satisfied and that 
$x$ and $c$ are strictly interior points for (1b). 
The global minimizer of this minimization problem is known as the
analytic center of the feasible region, and may be viewed as a feasible 
point that is as far from the boundary of the constraints as possible.
Note that terms in the above sumations corresponding to infinite bounds are
ignored, and that equality constraints are treated specially.
Appropriate "primal" Newton corrections are used to generate a sequence
of improving points converging to the analytic center, while the iteration
is stabilized by performing inesearches along these corrections with respect to
$\phi (x,\;c)$.

In order to make the solution as efficient as possible, the 
variables and constraints are reordered internally by the package 
``QPP`` prior to solution. In particular, fixed variables, and 
free (unbounded on both sides) constraints are temporarily removed.

references
----------

The basic algorithm is that of

  Y. Zhang,
  ``On the convergence of a class of infeasible interior-point methods 
  for the horizontal linear complementarity problem''.
  *SIAM J. Optimization* **4(1)** (1994) 208-227,

with a number of enhancements described by

  A. R. Conn, N. I. M. Gould, D. Orban and Ph. L. Toint,
  ``A primal-dual trust-region algorithm for minimizing a non-convex 
  function subject to general inequality and linear equality constraints''.
  *Mathematical Programming **87** (1999) 215-249.

