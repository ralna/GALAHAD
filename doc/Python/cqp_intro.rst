purpose
-------

The ``cqp`` package uses a **primal-dual interior-point method** to solve a
given **convex quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
or the **shifted-least-distance** objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric postive-semi-definite and $m$ by $n$ matrices,  
$g$, $w$ and $x^0$ are vectors, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/cqp.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c\;\;\mbox{(1a)}$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(1b)}$$
the **dual optimality conditions**
$$H x + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; 
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

In the shifted-least-distance case, $g$ is shifted by $-W^2 x^0$,
and $H = W^2$, where $W$ is the diagonal matrix whose entries are the $w_j$.

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

The method aims at each stage to reduce the
overall violation of (1a), (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; c, \; y, \; y^{l}, \; y^{u}, \; z, \; z^{l}, \; z^{u})$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \; c, \; y, \; y^{l}, \; 
y^{u} ,\;z,\;z^{l} ,\;z^{u} )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (1a), (2a) and a parameterized ``residual
trajectory'' perturbation of (3); residual trajectories
proposed by Zhang (1994) and Zhao and Sun (1999) are possibilities.
An improved estimate $v + \alpha \Delta v$
is then used, where the step-size $\alpha$
is chosen as close to 1.0 as possible while ensuring both that
(1b) and (2b) continue to hold and that the individual components
which make up the complementary slackness
(3) do not deviate too significantly
from their average value. The parameter that controls the perturbation
of (3) is ultimately driven to zero.

If the algorithm believes that it is close to the solution, it may take a
speculative ``pounce'' extrapolation, based on an estimate of the ultimate 
active set, to avoid further costly iterations. If the pounce is unsuccessful, 
the iteration continues, but further pounces may be attempted later.

The Newton equations are solved  by applying the matrix factorization 
package ``SBLS``, but there are options
to factorize the matrix as a whole (the so-called "augmented system"
approach), to perform a block elimination first (the "Schur-complement"
approach), or to let the method itself decide which of the two
previous options is more appropriate.
The "Schur-complement" approach is usually to be preferred when all the
weights are nonzero or when every variable is bounded (at least one side),
but may be inefficient if any of the columns of $A$ is too dense.

Optionally, the problem may be pre-processed temporarily to eliminate dependent
constraints using the package ``FDC``. This may improve the
performance of the subsequent iteration.

references
----------

The basic algorithm is a generalisation of those of

  Y. Zhang,
  ``On the convergence of a class of infeasible interior-point methods 
  for the horizontal linear complementarity problem''.
  *SIAM J. Optimization* **4(1)** (1994) 208-227,

and 

  G. Zhao and J. Sun,
  ``On the rate of local convergence of high-order infeasible 
  path-following algorithms for $P_*$ linear complementarity problems''.
  *Computational Optimization and Applications* **14(1)* (1999) 293-307,

with many enhancements described by

  N. I. M. Gould, D. Orban and D. P. Robinson,
  ``Trajectory-following methods for large-scale degenerate 
  convex quadratic programming'',
  *Mathematical Programming Computation* **5(2)** (2013) 113-142.

