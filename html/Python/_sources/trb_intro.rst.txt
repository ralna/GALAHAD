purpose
-------

The ``trb`` package uses a trust-region method to find a (local)
minimizer of a differentiable objective function $f(x)$ of
many variables $x$, where the variables satisfy the simple 
bounds $x^l <= x <= x^u$.  The method offers the choice of
direct and iterative solution of the key subproblems, and
is most suitable for large problems. First derivatives are required,
and if second derivatives can be calculated, they will be exploited.

See Section 4 of $GALAHAD/doc/trb.pdf for additional details.

method
------

A trust-region method is used. In this, an improvement to a current
estimate of the required minimizer, $x_k$ is sought by computing a
step $s_k$. The step is chosen to approximately minimize a model $m_k(s)$
of $f(x_k + s)$ within the intersection of the bound constraints
$x^l \leq x \leq x^u$ and a trust region $\|s_k\| \leq \Delta_k$
for some specified positive "radius" $\Delta_k$. The quality of the
resulting step $s_k$ is assessed by computing the "ratio"
$(f(x_k) - f(x_k + s_k))/ (m_k(0) - m_k(s_k))$.
The step is deemed to have succeeded if the ratio exceeds a given $\eta_s > 0$,
and in this case $x_{k+1} = x_k + s_k$. Otherwise
$x_{k+1} = x_k$, and the radius is reduced by powers of a given
reduction factor until it is smaller than $\|s_k\|$. If the ratio
is larger than  $\eta_v \geq \eta_d$, the radius will be increased so that
it exceeds $\|s_k\|$ by a given increase factor. The method will terminate
as soon as $\|\nabla_x f(x_k)\|$ is smaller than a specified value.

Either linear or quadratic models $m_k(s)$ may be used. The former will be
taken as the first two terms $f(x_k) + s^T \nabla_x f(x_k)$
of a Taylor series about $x_k$, while the latter uses an
approximation to the first three terms
$f(x_k) + s^T \nabla_x f(x_k) + \frac{1}{2} s^T B_k s$,
for which $B_k$ is a symmetric approximation to the Hessian
$\nabla_{xx} f(x_k)$; possible approximations include the true Hessian,
limited-memory secant and sparsity approximations and a scaled identity matrix.
Normally a two-norm trust region will be used, but this may change
if preconditioning is employed.

The model minimization is carried out in two stages.
Firstly, the so-called generalized Cauchy point for the quadratic
subproblem is found---the purpose of this point is to ensure that the
algorithm converges and that the set of bounds which are satisfied as
equations at the solution is rapidly identified.  Thereafter an
improvement to the quadratic model on the face of variables predicted
to be active by the Cauchy point is sought using either a
direct approach involving factorization or an
iterative (conjugate-gradient/Lanczos) approach based on approximations
to the required solution from a so-called Krlov subspace. The direct
approach is based on the knowledge that the required solution
satisfies the linear system of equations $(B_k + \lambda_k I) s_k
= - \nabla_x f(x_k)$, involving a scalar Lagrange multiplier $\lambda_k$,
on the space of inactive variables.
This multiplier is found by uni-variate root finding, using a safeguarded
Newton-like process, by ``TRS`` or ``DPS``
(depending on the norm chosen). The iterative approach
uses ``GLTR``, and is best accelerated by preconditioning
with good approximations to $B_k$ using ``PSLS``. The
iterative approach has the advantage that only matrix-vector products
$B_k v$ are required, and thus $B_k$ is not required explicitly.
However when factorizations of $B_k$ are possible, the direct approach
is often more efficient.

The iteration is terminated as soon as the Euclidean norm of the 
projected gradient,
$$\|\min(\max( x_k - \nabla_x f(x_k), x^l), x^u) -x_k\|_2,$$
is sufficiently small. At such a point, $\nabla_x f(x_k) = z_k$, 
where the $i$-th dual variable $z_i$ is non-negative if $x_i$ is on its
lower bound $x^l_i$, non-positive if $x_i$ is on its upper bound $x^u_i$,
and zero if $x_i$ lies strictly between its bounds.

reference
---------

The generic bound-constrained trust-region method is described in detail in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint,
  Trust-region methods.
  SIAM/MPS Series on Optimization (2000).
