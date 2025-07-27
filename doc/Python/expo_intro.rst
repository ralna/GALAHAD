purpose
-------

The ``expo`` package uses an **exponential-penalty function method** 
to solve a given **constrained optimization problem**.
The aim is to find a (local) minimizer of a differentiable 
**objective function** $f(x)$ of $n$ variables $x$, subject 
to $m$ **general constraints** $c_l \leq c(x) \leq c_u$
and **simple-bound constraints** $x_l \leq x \leq x_u$ on the variables.
Here, any of the components of the vectors of bounds 
$c_l$, $c_u$, $x_l$ and $x_u$ may be infinite. 
The method offers the choice of direct and iterative solution 
of the key unconstrained-optimization subproblems, and
is most suitable for large problems. First derivatives are required,
and if second derivatives can be calculated, they will be exploited---if
the product of second derivatives with a vector may be found but
not the derivatives themselves, that may also be exploited.

**N.B. This package is currently a beta release, and aspects may change
before it is formally released**

See Section 4 of $GALAHAD/doc/expo.pdf for additional details.

terminology
-----------

The **exponential penalty function** is defined to be
$$\begin{array}{rl}\phi(x,w,\mu,v,\nu) \!\! & = f(x) 
+ \sum_{i} \mu_{li} w_{li} \exp[(c_{li} - c_i(x))/\mu_{li}] \\
& \;\;\;\;\;\;\;\;\;\;\;\;\; 
+ \sum_{i} \mu_{ui} w_{ui} \exp[(c_i(x) - c_{ui})/\mu_{ui}] \\
& \;\;\;\;\;\;\;\;\;\;\;\;\; 
+ \sum_{j} \nu_{lj} v_{lj} \exp[(x_{lj} - x_j)/\nu_{lj}] \\
& \;\;\;\;\;\;\;\;\;\;\;\;\; 
+ \sum_{j} \nu_{uj} v_{uj} \exp[(x_j - x_{uj})/\nu_{uj}],
\end{array}
$$
where $c_{li}$, $c_{ui}$ and $c_i(x)$ are the $i$-th components
of $c_l$, $c_u$ and $c(x)$, and 
$c_{lj}$, $c_{uj}$ and $x_j$ are the $j$-th components
of $x_l$, $x_u$ and $x$, respectively. Here the components
of $\mu_l$, $\mu_u$, $\nu_l$ and $\nu_u$ 
are separate **penalty parameters** for each lower and upper, 
general and simple-bound constraint, respectively,
while those of $w_l$, $w_u$, $v_l$, $v_u$
are likewise separate **weights** for the same. The algorithm iterates by
approximately minimizing $\phi(x,w,\mu,v,\nu)$
for a fixed set of penalty parameters and weights, and then adjusting 
these parameters and weights. The adjustments are designed so the
sequence of approximate minimizers of $\phi$ converge
to that of the specified constrained optimization problem.

Key constructs are the **gradient** of the objective function
$$g(x) := \nabla_x f(x),$$
the **Jacobian** of the vector of constraints, 
$$J(x) := \nabla_x c(x),$$
and the **gradient** and **Hessian** of the Lagrangian function
$$g_L(x,y,z) := g(x) - J^T(x)y - z  \;\;\mbox{and}\;\;
H_L(x,y) := \nabla_{xx}  \left[ f(x) - \sum_{i} y_i c_i(x)\right]$$
for given vectors $y$ and $z$. 

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$c_l \leq c(x) \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(1)}$$
the **dual optimality conditions**
$$g(x) = J^{T}(x) y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; 
z = z_l + z_u,\;\;\mbox{(2a)}$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; 
z_u \leq 0,\;\;\mbox{(2b)}$$
and the **complementary slackness conditions**
$$( c(x) - c_l )^{T} y_l = 0,\;\; ( c(x) - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,\;\;\mbox{(3)}$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general constraints, and the **dual variables** for the simple bounds,
respectively, and where the vector inequalities hold component-wise.

method
------
The method employed involves a sequential minimization of the exponential
penalty function $\phi(x,w,\mu,v,\nu)$ for a sequence of positive penalty 
parameters $(\mu_{lk}, \mu_{uk}, \nu_{lk}, \nu_{uk})$ 
and weights $(w_{lk}, w_{uk}, v_{lk}, v_{uk})$,
for increasing $k \geq 0$. Convergence is ensured if the
penalty parameters are forced to zero, and may be accelerated
by adjusting the weights. The minimization of $\phi(x,w,\mu,v,\nu)$ 
is  accomplished using the trust-region unconstrained solver 
``TRU``. Although critical points $\{x_k\}$ of 
$\phi(x,w_k,\mu_k,v_k,\nu_k)$ converge to a local solution $x_*$
of the underlying problem, the reduction of the penalty parameters to
zero often results in $x_k$ being a poor starting point for the minimization 
of $\phi(x,w_{k+1},\mu_{k+1},v_{k+1},\nu_{k+1})$. Consequently, 
a careful extrapolated starting point from $x_k$ is used instead. Moreover,
once the algorithm is confident that it is sufficiently close to $x_*$,
it switches to Newton's method to accelerate the convergence. Both the
extrapolation and the Newton iteration rely on the block-linear-system
solver ``SSLS``.

The iteration is terminated as soon as residuals to the
optimality conditions (1)--(3) are sufficiently small. For 
infeasible problems, this will not be possible, and instead the
residuals to (1) will be made as small as possible.

references
----------

The method is described in detail in

  N.\ Gould, S.\ Leyffer, A.\ Montoison and C.\ Vanaret (2025)
  The exponential multiplier method in the 21st century.
  RAL Technical Report, in preparation.
