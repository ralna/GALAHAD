purpose
-------

The ``sllsb`` package uses a **primal-dual interior-point crossover method** 
to solve a given **simplex-constrained linear least-squares problem**.
The aim is to minimize the (regularized) least-squares objective function
$$q(x) = \frac{1}{2} \| A_o x - b\|_W^2 +  \frac{1}{2}\sigma \|x-x_s\|_2^2$$ 
where the variables $x$ are required to lie within the
**regular simplex**
$$e^T x = 1 \;\;\mbox{and}\;\; x \geq 0,$$
or an intersection of **multiple non-overlapping regular simplices**
$$e_{\cal C_i}^T x_{\cal C_i}^{} = 1 \;\;\mbox{and}\;\; x_{\cal C_i}^{}
\geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,$$
where the $o$ by $n$ real **design matrix** $A_o$, 
the vector $b$ of **observations**,
the **shifts** $x_s$, 
and the non-negative **weights** $w$ and $\sigma$ are given, 
$e$ is the vector of ones, 
the vector $v_{\cal C}$ is made up of those entries of $v$ indexed by
the set $\cal C$, 
the index sets of **cohorts** $\cal C_i \subseteq \{1,\ldots,n\}$
for which $\cal C_i \cap \cal C_j = \emptyset$ for $1 \leq i, \neq j \leq m$,
and where the Euclidean and weighted-Euclidean norms
are given by $\|v\|_2^2 = v^T v$ and $\|v\|_W^2 = v^T W v$,
respectively, with $W = \mbox{diag}(w)$.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/sllsb.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$e_{\cal C_i}^T x_{\cal C_i}^{} = 1,\;\;\mbox{(1a)}$$
and
$$x_{\cal C_i} \geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,\;\;\mbox{(1b)}$$
the **dual optimality conditions**
$$A_o^T W ( A_o x - b) + \sigma (x-x_s)  = \sum_{i=1}^m e_{\cal C_i} y_i + z,\;\;\mbox{(2a)}$$
and
$$z_{\cal C_i} \geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,\;\;\mbox{(2b)}$$
and the **complementary slackness conditions**
$$x^T z = 0,\;\;\mbox{(3)}$$
where the components of the vector $y$ are known as the 
**Lagrange multipliers** for the constraints (1a),
$z$ is are **dual variables** for the bounds (1b),
and the vector inequalities hold component-wise.

method
------

Primal-dual interior point methods iterate towards a point that satisfies 
these optimality conditions by ultimately aiming to satisfy
(1a), (2a) and (3), while ensuring that (1b) and (2b) are
satisfied as strict inequalities at each stage.
Appropriate norms of the amounts by
which (2a) and (2b) fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. The fact that (1b) and (2b) are satisfied as strict
inequalities gives such methods their other title, namely
interior-point methods.

The method aims at each stage to reduce the
overall violation of (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; z)$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \;z, )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (2a) and a parameterized ''residual
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
speculative ''pounce'', based on an estimate of the ultimate active set, 
to avoid further costly iterations. If the pounce is unsuccessful, the 
iteration continues, and further pounces may be attempted later.

The Newton equations are solved  by applying the matrix factorization 
package ``SLS``.
All computation is designed to exploit sparsity in $A_o$.

reference
---------

The basic algorithm is a generalisation of those of

  Y. Zhang,
  ''On the convergence of a class of infeasible interior-point methods 
  for the horizontal linear complementarity problem''.
  *SIAM J. Optimization* **4(1)** (1994) 208-227,

and 

  G. Zhao and J. Sun,
  ''On the rate of local convergence of high-order infeasible 
  path-following algorithms for $P_*$ linear complementarity problems''.
  *Computational Optimization and Applications* **14(1)** (1999) 293-307,

with many enhancements described by

  N. I. M. Gould, D. Orban and D. P. Robinson,
  ''Trajectory-following methods for large-scale degenerate 
  convex quadratic programming'',
  *Mathematical Programming Computation* **5(2)** (2013) 113-142

and tailored for a regularized linear least-squares objective.
