purpose
-------

The ``arc`` package uses a regularization method to find a (local)
minimizer of a differentiable objective function $f(x)$ of
many variables $x$. The method offers the choice of direct
and iterative solution of the key subproblems, and is most
suitable for large problems. First derivatives are required, and
if second derivatives can be calculated, they will be exploited.

See Section 4 of $GALAHAD/doc/arc.pdf for additional details.

method
------

An adaptive cubic regularization method is used.
In this, an improvement to a current
estimate of the required minimizer, $x_k$ is sought by computing a
step $s_k$. The step is chosen to approximately minimize a model $m_k(s)$
of $f(x_k + s)$ that includes a weighted term $\sigma_k \|s_k\|^3$
for some specified positive weight $\sigma_k$. The quality of the
resulting step $s_k$ is assessed by computing the "ratio"
$(f(x_k) - f(x_k + s_k))/ (m_k(0) - m_k(s_k))$.
The step is deemed to have succeeded if the ratio exceeds a given $\eta_s > 0$,
and in this case $x_{k+1} = x_k + s_k$. Otherwise
$x_{k+1} = x_k$, and the weight is increased by powers of a given
increase factor up to a given limit. If the ratio is larger than
$\eta_v \geq \eta_d$, the weight will be decreased by powers of a given
decrease factor again up to a given limit. The method will terminate
as soon as $\|\nabla_x f(x_k)\|$ is smaller than a specified value.

Either linear or quadratic models $m_k(s)$ may be used. The former will be
taken as the first two terms $f(x_k) + s^T \nabla_x f(x_k)$
of a Taylor series about $x_k$, while the latter uses an
approximation to the first three terms
$f(x_k) + s^T \nabla_x f(x_k) + \frac{1}{2} s^T B_k s$,
for which $B_k$ is a symmetric approximation to the Hessian
$\nabla_{xx} f(x_k)$; possible approximations include the true Hessian,
limited-memory secant and sparsity approximations and a scaled identity matrix.
Normally a two-norm regularization will be used, but this may change
if preconditioning is employed.

An approximate minimizer of the cubic model
is found using either a direct approach involving factorization or an
iterative (conjugate-gradient/Lanczos) approach based on approximations
to the required solution from a so-called Krlov subspace. The direct
approach is based on the knowledge that the required solution
satisfies the linear system of equations $(B_k + \lambda_k I) s_k
= - \nabla_x f(x_k)$ involving a scalar Lagrange multiplier $\lambda_k$.
This multiplier is found by uni-variate root finding, using a safeguarded
Newton-like process, by ``RQS`` or ``DPS``
(depending on the norm chosen). The iterative approach
uses ``GLRT``, and is best accelerated by preconditioning
with good approximations to $B_k$ using ``PSLS``. The
iterative approach has the advantage that only matrix-vector products
$B_k v$ are required, and thus $B_k$ is not required explicitly.
However when factorizations of $B_k$ are possible, the direct approach
is often more efficient.


references
----------

The generic adaptive cubic regularization method is described in detail in

  C. Cartis,  N. I. M. Gould and Ph. L. Toint,
  ``Adaptive cubic regularisation methods for unconstrained optimization.
  Part I: motivation, convergence and numerical results''
  *Mathematical Programming* **127(2)** (2011) 245--295,

and uses ``tricks'' as suggested in

  N. I. M. Gould, M. Porcelli and Ph. L. Toint,
  ``Updating the regularization parameter in the adaptive cubic regularization
  algorithm''.
  *Computational Optimization and Applications*
  **53(1)** (2012) 1-22.
