purpose
-------

The ``bqp`` package uses a **preconditioned, projected-gradient method** to 
solve a given **bound-constrained convex quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
subject to the simple bounds
$$x_l \leq x \leq x_u,$$
where $H$ is a given $n$ by $n$ symmetric postive-semi-definite matrix,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/bqp.pdf for a brief description of the
method employed and other details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$H x + g = z, \;\; z = z_l + z_u, z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,$$
and the **complementary slackness conditions**
$$(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vector $z$ is known as the **dual variables** for the bounds,
and where the vector inequalities hold component-wise.

method
------

Projected-gradient methods iterate towards a point
that satisfies these conditions by ultimately aiming to satisfy
$H x + g = z$ and $z = z_l + z_u$, while satifying the remaining
optimality conditions at each stage. Appropriate norms of the amounts by 
which the optimality conditions fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. 

The method is iterative. Each iteration proceeds in two stages.
Firstly, the so-called generalized Cauchy point for the quadratic
objective is found.  (The purpose of this point is to ensure that the
algorithm converges and that the set of bounds which are satisfied as
equations at the solution is rapidly identified.)  Thereafter an
improvement to the objective is sought using either a
direct-matrix or truncated conjugate-gradient algorithm.

reference
---------

This is a specialised version of the method presented in 

  A. R. Conn, N. I. M. Gould and Ph. L. Toint,
  Global convergence of a class of trust region algorithms
  for optimization with simple bounds.
  *SIAM Journal on Numerical Analysis* **25** (1988) 433-460.
