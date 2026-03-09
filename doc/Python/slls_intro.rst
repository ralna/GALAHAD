purpose
-------

The ``slls`` package uses a **preconditioned project-gradient method** to solve
a given **simplex-constrained linear least-squares problem**.
The aim is to minimize the (regularized) least-squares objective function
$$q(x) = \frac{1}{2} \| A_o x - b\|_W^2 +  \frac{1}{2}\sigma \|x-x_s\|_2^2$$ 
where the variables $x$ are required to lie within the
**regular simplex**
$$e^T x = 1 \;\;\mbox{and}\;\; x \geq 0,$$
or an intersection of **multiple non-overlapping regular simplices**
$$e_{\cal C_i}^T x_{\cal C_i}^{} = 1 \;\;\mbox{and}\;\; x_{\cal C_i}^{}
\geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,$$
where the $o$ by $n$ real **design matrix** $A_o$, the vector $b$
of **observations**
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

See Section 4 of $GALAHAD/doc/slls.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$e_{\cal C_i}^T x_{\cal C_i}^{} = 1 \;\;\mbox{and}\;\; x_{\cal C_i}^{}
\geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m$$
the **dual optimality conditions**
$$A_o^T W ( A_o x - b) + \sigma (x-x_s) = \sum_{i=1}^m e_{\cal C_i} y_i + z,
\;\;\mbox{and}\;\; z_{\cal C_i} \geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,$$
and the **complementary slackness conditions**
$$x^T z = 0,$$
where the components of the vector $y$ are known as the 
**Lagrange multipliers** for the primal equality constraints
$z$ is are **dual variables** for the primal inequalities,
and the vector inequalities hold component-wise.

for some scalar Lagrange multiplier $\lambda$, where the 
vector $z$ is known as the **dual variables** for the bounds $x \geq 0$,
and the vector inequalities hold component-wise.

method
------

Projected-gradient methods iterate towards a point that satisfies these 
optimality conditions by ultimately aiming to satisfy the dual optimality 
condistions, while ensuring that the remaining conditions are satisfied 
at each stage. Appropriate norms of the amounts by which the optimality 
conditions fail to be satisfied are known as the primal and dual infeasibility, 
and the violation of complementary slackness, respectively.

The method is iterative. Each iteration proceeds in two stages.
Firstly, a search direction $s$ from the current estimate of the solution
$x$ is computed. This may be in a scaled steepest-descent direction, or,
if the working set of variables on bounds has not changed dramatically,
in a direction that provides an approximate minimizer of the objective
over a subspace comprising the currently free-variables. The latter is
computed either using an appropriate sparse factorization by the
galahad package SBLS, or by the
conjugate-gradient least-squares (CGLS) method; tt may be necessary to
regularize the subproblem very slightly to avoid a ill-posedness.
Thereafter, a piecewise linesearch (arc search) is carried out along
the arc $x(\alpha) = P( x + \alpha s)$ for $\alpha > 0$,
where the projection operator $P(v)$ gives the nearest point to $v$ 
within the intersection of specified simplices;
thus this arc bends the search direction into the feasible region.
The arc search is performed either exactly, by passing through a set
of increasing breakpoints at which it changes direction, or inexactly,
by evaluating a sequence of different $\alpha$  on the arc.
All computation is designed to exploit sparsity in $A_o$.

reference
---------

Full details are provided in

  H. Al Daas, J. M. Fowkes, N. I. M. Gould and J. Huntley (2026).
  ''Linear least-squares over the unit simplex''.
  STFC-Rutherford Appleton Laboratory Computational Mathematics Group
  Internal Report 2026-1.
