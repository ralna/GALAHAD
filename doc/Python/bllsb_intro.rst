purpose
-------

The ``bllsb`` package uses a **primal-dual interior-point crossover method** 
to solve a **bound-constrained linear least-squares problem**,
The aim is to minimize the (regularized) least-squares objective function
$$q(x) = \frac{1}{2} \| A_o x - b\|_W^2 +  \frac{1}{2}\sigma \|x\|^2$$ 
subject to the simple bounds
$$x_l \leq x \leq x_u,$$
where the norms $\|r\|_W = \sqrt{\sum_{i=1}^o w_i r_i^2}$
and $\|x\| = \sqrt{\sum_{i=1}^n x_i^2}$,
$A_o$ is a given  $o$ by $n$ matrix,
$b$, and $w$ are vectors, $\sigma \geq 0$ is a scalar, 
and any of the components 
of the vectors $x_l$ or $x_u$ may be infinite.
The method is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/bllsb.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$x_l \leq x \leq x_u,\;\;\mbox{(1)}$$
the **dual optimality conditions**
$$A_o^T W ( A_o x-b ) + \sigma x = z,\;\;  
\mbox{and}\;\; z = z_l + z_u,\;\;\mbox{(2a)}$$
and
$$z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,\;\;\mbox{(2b)}$$
and the **complementary slackness conditions**
$$(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,\;\;\mbox{(3)}$$
where the components of the vector $z$ are known as 
the **dual variables** for the bounds
where the vector inequalities hold component-wise,
and where $W$ is the diagonal matrix whose entries are the $w_j$.

method
------

Primal-dual interior point methods iterate towards a point that satisfies 
these optimality conditions by ultimately aiming to satisfy
(2a) and (3), while ensuring that (1) and (2b) are
satisfied as strict inequalities at each stage.
Appropriate norms of the amounts by
which (2a) and (3) fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. The fact that (1) and (2b) are satisfied as strict
inequalities gives such methods their other title, namely
interior-point methods.

The method aims at each stage to reduce the
overall violation of (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; z, \; z^{l}, \; z^{u})$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \;z,, \;z^{l} ,\;z^{u} )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (2a) and a parameterized ``residual
trajectory'' perturbation of (3); residual trajectories
proposed by Zhang (1994) and Zhao and Sun (1999) are possibilities.
An improved estimate $v + \alpha \Delta v$
is then used, where the step-size $\alpha$
is chosen as close to 1.0 as possible while ensuring both that
(1) and (2b) continue to hold and that the individual components
which make up the complementary slackness
(3) do not deviate too significantly
from their average value. The parameter that controls the perturbation
of (3) is ultimately driven to zero.

If the algorithm believes that it is close to the solution, it may take a
speculative ``pounce'', based on an estimate of the ultimate active set, 
to avoid further costly iterations. If the pounce is unsuccessful, the 
iteration continues, and further pounces may be attempted later.

The Newton equations are solved  by applying the matrix factorization 
package ``SLS``.

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
  *Mathematical Programming Computation* **5(2)** (2013) 113-142

and tailored for a regularized linear least-squares objective.
