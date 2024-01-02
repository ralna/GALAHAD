purpose
-------

The ``blls`` package uses a **preconditioned project-gradient method** to solve
a given **bound-constrained linear least-squares problem**.
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
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/blls.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$A_o^T W( A_o x - b) + \sigma x = z, \;\; z = z_l + z_u, z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,$$
and the **complementary slackness conditions**
$$(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vector $z$ is known as the **dual variables** for the bounds,
and where the vector inequalities hold component-wise.

method
------

Projected-gradient methods iterate towards a point that satisfies 
these optimality conditions by ultimately aiming to satisfy
$A_o^T W ( A_o x - b) + \sigma x = z$, while ensuring that the remaining 
conditions are satisfied at each stage.
Appropriate norms of the amounts by which the optimality conditions
fail to be satisfied are known as the primal and dual infeasibility, 
and the violation of complementary slackness, respectively.

The method is iterative. Each iteration proceeds in two stages.
Firstly, a search direction $s$ from the current estimate of the solution
$x$ is computed. This may be in a scaled steepest-descent direction, or,
if the working set of variables on bounds has not changed dramatically,
in a direction that provides an approximate minimizer of the objective
over a subspace comprising the currently free-variables. The latter is
computed either using an appropriate sparse factorization by the
package ``SBLS``, or by the
conjugate-gradient least-squares (CGLS) method; tt may be necessary to
regularize the subproblem very slightly to avoid a ill-posedness.
Thereafter, a piecewise linesearch (arc search) is carried out along
the arc $x(\alpha) = P( x + \alpha s)$ for $\alpha > 0$,
where the projection operator
is defined component-wise at any feasible point $v$ to be
$$P_j(v) = \min( \max( x_j, x_j^{l}), x_j^{u});$$
thus this arc bends the search direction into the feasible region.
The arc search is performed either exactly, by passing through a set
of increasing breakpoints at which it changes direction, or inexactly,
by evaluating a sequence of different $\alpha$  on the arc.
All computation is designed to exploit sparsity in $A_o$.

reference
---------

Full details are provided in

  N. I. M. Gould (2022).
  ``A projection method for bound-constrained linear least-squares''.
  STFC-Rutherford Appleton Laboratory Computational Mathematics Group
  Internal Report 2023-1 (2023).
