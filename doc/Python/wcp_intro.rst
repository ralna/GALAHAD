purpose
-------

The ``wcp`` package uses a **primal-dual interior-point method** to find a 
well-centered point within a **polyhedral set**.
The aim is to find a point that lies interior to the boundary of the 
polyhedron definied by the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $A$ is a given $m$ by $n$ matrix, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$, since full advantage is taken of
"any zero coefficients in the matrix $A$.
The package identifies infeasible problems, and problems for which there is 
no strict interior.

See Section 4 of $GALAHAD/doc/wcp.pdf for a brief description of the
method employed and other details.

terminolgy
----------

More specifically, if possible, the package finds a solution to the 
system of  **primal optimality equations**
$$A x = c,\;\;\mbox{(1)}$$
the **dual optimality equations**
$$g = A^{T} y + z,\;\; y = y^{l} + y^{u} \;\;\mbox{and}\;\; 
z = z^{l} + z^{u},\;\;\mbox{(2)}$$
and **perturbed complementary slackness equations**
$$( c_i^{} - c^l_i ) y^{l}_i = (\mu_c^{l})_i^{} \;\;\mbox{and}\;\; 
( c_i^{} - c_i^u ) y^{u}_i = (\mu_c^{u})_i^{}, \;\;\; 
i = 1, \ldots , m,\;\;\mbox{(3)}$$
and
$$((x^{}_j - x^l_j ) z_j^{l} = (\mu_x^{l})_j^{} \;\;\mbox{and}\;\; 
( x^{}_j - x^u_j ) z_j^{u} = (\mu_x^{u})_j^{}, \;\;\;  
j = 1, \ldots , n,\;\;\mbox{(4)}$$
for which
$$c^{l} \leq c \leq c^{u}, \;\; x^{l} \leq x \leq x^{u}, \;\;  y^{l} \geq 0, 
\;\;  y^{u} \leq 0, \;\;  z^{l} \geq 0 \;\; \mbox{and} \;\;  z^{u} \leq 0.
\;\;\mbox{(5)}$$
Here $\mu_c^{l}$, $\mu_c^{u}$, $\mu_x^{l}$ and $\mu_x^{u}$ are
vectors of strictly positive **targets**, $g$ is another given 
target vector (which is often zero),
$(y^{l}, y^{u})$ and $(z^{l}, z^{u})$ are **Lagrange multipliers**
for the linear constraints and **dual variables** for the simple bounds 
respectively, and vector inequalities hold component-wise;  
$c$ gives the constraint value $A x$.
Since the perturbed complementarity equations normally imply that
$$c^{l} < c < c^{u}, \;\; x^{l} < x < x^{u}, \;\;  y^{l} > 0, \;\;  y^{u} < 0,
\;\;  z^{l} > 0 \;\; \mbox{and} \;\;  z^{u} < 0,
\;\;\mbox{(6)}$$
such a primal-dual point $(x, c, y^{l}, y^{u}, z^{l}, z^{l})$
may be used, for example, as a feasible starting point for primal-dual 
interior-point methods for solving the linear programming problem 
of minimizing $g^T x$ within the given polyhedral set.
method

method
------

The algorithm is iterative, and at each major iteration attempts
to find a solution to the perturbed system (1), (2),
$$( c_i^{} - c^l_i + (\theta_c^l)_i^{} ) ( y^{l}_i + (\theta_y^l)_i^{} )
= (\mu_c^{l})_i^{} \;\;\mbox{and}\;\;
( c_i^{} - c_i^u - (\theta_c^u)_i^{} ) y^{u}_i - (\theta_y^u)_i^{} )
= (\mu_c^{u})_i^{}, \;\;\; i = 1, \ldots , m,\;\;\mbox{(7)}$$
$$
( x_j^{} - x^l_j + (\theta_x^l)_j^{} ) ( z^{l}_j + (\theta_z^l)_j^{} )
= (\mu_x^{l})_j^{} \;\;\mbox{and}\;\;
( x_j^{} - x_j^u - (\theta_x^u)_j^{} ) ( z^{u}_j - (\theta_z^u)_j^{} )
= (\mu_x^{u})_j^{}, \;\;\;
j = 1, \ldots , n,\;\;\mbox{(8)}$$
and
$$c^{l} - \theta_c^l < c < c^{u} + \theta_c^u, \;\;
x^{l} - \theta_x^l < x < x^{u} + \theta_x^u, \;\;
y^{l} > - \theta_y^l , \;\;
y^{u} < \theta_y^u , \;\;
z^{l} > - \theta_z^l \;\; \mbox{and} \;\;
z^{u} < \theta_z^u ,\;\;\mbox{(9)}$$
where the vectors of perturbations 
$\theta^l_c$, $\theta^u_c$, $\theta^l_x$, $\theta^u_x$, $\theta^l_x$,
$\theta^u_x$, $\theta^l_y$, $\theta^u_y$, $\theta^l_z$ and $\theta^u_z$,
are non-negative. Rather than solve (1), (2) and (7)--(9)
exactly, we instead seek a feasible point for the easier relaxation
(1), (2) and
$$\begin{array}{cccccl}
\gamma (\mu_c^{l})_i^{} & \leq &
( c_i^{} - c^l_i + (\theta_c^l)_i^{} ) ( y^{l}_i + (\theta_y^l)_i^{} )
& \leq & (\mu_c^{l})_i^{} / \gamma & \;\;\mbox{and}\;\; \\
\gamma (\mu_c^{u})_i^{} & \leq &
( c_i^{} - c_i^u - (\theta_c^u)_i^{} ) ( y^{u}_i - (\theta_y^u)_i^{} )
& \leq & (\mu_c^{u})_i^{}, /\gamma &
\;\;\;  i = 1, \ldots , m, \;\;\mbox{and}\;\; \\
\gamma (\mu_x^{l})_j^{} & \leq &
( x_j^{} - x^l_j + (\theta_x^l)_j^{} ) ( z^{l}_j + (\theta_z^l)_j^{} )
& \leq & (\mu_x^{l})_j^{} /\gamma & \;\;\mbox{and}\;\; \\
\gamma (\mu_x^{u})_j^{}  & \leq &
( x_j^{} - x_j^u - (\theta_x^u)_j^{} ) 
( z^{u}_j - (\theta_z^u)_j^{} ) 
& \leq & (\mu_x^{u})_j^{} /\gamma , & \;\;\; j = 1, \ldots , n,
\end{array} \;\;\mbox{(10)}$$
for some $\gamma \in (0,1]$ which is allowed to be smaller than one 
if there is a nonzero perturbation.

Given any solution to (1)--(2) and (10) satisfying (9),
the perturbations are reduced (sometimes to zero) so as to ensure that the 
current solution is feasible for the next perturbed problem. Specifically,
the perturbation $(\theta^l_c)_i^{}$ for the constraint $c_i^{} \geq c^l_i$
is set to zero if $c_i$ is larger than some given parameter $\epsilon > 0$.
If not, but $c_i$ is strictly positive, the perturbation will be
reduced by a multiplier $\rho \in (0,1)$. Otherwise, the new perturbation 
will be set to $\xi (\theta^l_c)_i^{} + ( 1 - \xi ) ( c_i^l - c_i^{} )$
for some factor $\xi \in (0,1)$. Identical rules are used to reduce the
remaining primal and dual perturbations. 
The targets $\mu_c^{l}$, $\mu_c^{u}$, $\mu_x^{l}$ and $\mu_x^{u}$
will also be increased by the factor $\beta \geq 1$ for those
(primal and/or dual) variables with strictly 
positive perturbations so as to try to accelerate the convergence. 

Ultimately the intention is to drive all the perturbations to zero. 
It can be shown that if the original problem (1)--(4) and (6) has 
a solution, the perturbations will be zero after a finite 
number of major iterations. Equally, if there is no interior solution (6), 
the sets of (primal and dual) variables that are necessarily at (one of) their
bounds for all feasible points---we refer to these as *implicit*
equalities---will be identified, as will the possibility that there is
no point (interior or otherwise) in the primal and/or dual feasible regions.

Each major iteration requires the solution $v = (x,c,z^l,z^u,y^l,y^u)$
of the nonlinear system (1), (2) and (7)--(9)
for fixed perturbations, using a minor iteration. The minor iteration
uses a stabilized (predictor-corrector) Newton method, 
in which the arc 
$v(\alpha) = v + \alpha \dot{v} + \alpha^2 \ddot{v}$, $\alpha \in [0,1]$,
involving the standard Newton step $\dot{v}$
for the equations (1), (2), (7) and (8), optionally augmented by a 
corrector $\ddot{v}$ to account for the nonlinearity in (7) and (8),
is truncated so as to ensure that
$$( c_i^{}(\alpha) - c^l_i + (\theta_c^l)_i^{} ) 
( y^{l}_i(\alpha) + (\theta_y^l)_i^{} )
\geq \tau (\mu_c^{l})_i^{}, \;\;\; i = 1, \ldots , m,$$
$$( c_i^{}(\alpha) - c_i^u - (\theta_c^u)_i^{} ) 
( y^{u}_i(\alpha) - (\theta_y^u)_i^{} )
\geq \tau (\mu_c^{u})_i^{}, \;\;\; i = 1, \ldots , m,$$
$$( x_j^{}(\alpha) - x^l_j + (\theta_x^l)_j^{} ) 
( z^{l}_j(\alpha) + (\theta_z^l)_j^{} )
\geq \tau (\mu_x^{l})_j^{}, \;\;\; j = 1, \ldots , n,$$
and
$$( x_j^{}(\alpha) - x_j^u - (\theta_x^u)_j^{} ) 
( z^{u}_j(\alpha) - (\theta_z^u)_j^{} )
\geq \tau (\mu_x^{u})_j^{}, \;\;\; j = 1, \ldots , n,$$
for some $\tau \in (0,1)$, always holds, and also so that the norm
of the residuals to 
(1), (2), (7) and (8) is reduced as much as possible. 
The Newton and corrector systems are solved using a factorization of
the Jacobian of its defining functions (the so-called "augmented system"
approach) or of a reduced system in which some of the trivial equations are
eliminated (the "Schur-complement" approach).
The factors are obtained using the package ``SBLS``.

In order to make the solution as efficient as possible, the 
variables and constraints are reordered internally
by the package ``QPP`` prior to solution. 
In particular, fixed variables, and 
free (unbounded on both sides) constraints are temporarily removed.
In addition, an attempt to identify and remove linearly dependent
equality constraints may be made by factorizing
$$\left(\begin{array}{cc}\alpha I & A^T_{\cal E} \\ A^{}_{\cal E} & 0
\end{array}\right),$$ where $A_{\cal E}$ denotes the gradients of 
the equality constraints and $\alpha > 0$ is a given scaling factor,
using ``SBLS``, and examining small pivot blocks.

reference
---------

The basic algorithm, its convergence analysis and results of
numerical experiments are given in

  C. Cartis and N. I. M. Gould,
  Finding a point n the relative interior of a polyhedron.
  Technical Report TR-2006-016, Rutherford Appleton Laboratory (2006).
