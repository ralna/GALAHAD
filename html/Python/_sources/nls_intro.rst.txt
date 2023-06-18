purpose
-------

The ``nls`` package uses a regularization method to find a (local) unconstrained
minimizer of a differentiable weighted sum-of-squares objective function
$$f(x) :=
\frac{1}{2} \sum_{i=1}^m w_i c_i^2(x) \equiv \frac{1}{2} \|c(x)\|^2_W$$
of many variables $x$ involving positive weights $w_i$, $i=1,\ldots,m$.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for large problems.
First derivatives of the residual function $c(x)$ are required, and if
second derivatives of the $c_i(x)$ can be calculated, they may be exploited.

See Section 4 of $GALAHAD/doc/nls.pdf for additional details.

terminology
-----------

The **gradient** $\nabla_x f(x)$ of a function $f(x)$ is the vector
whose $i$-th component is $\partial f(x)/\partial x_i$.
The **Hessian** $\nabla_{xx} f(x)$ of $f(x)$ is the symmetric matrix
whose $i,j$-th entry is $\partial^2 f(x)/\partial x_i \partial x_j$.
The Hessian is **sparse** if a significant and useful proportion of the
entries are universally zero.

The algorithm used by the package is iterative. From the current best estimate
of the minimizer $x_k$, a trial improved point $x_k + s_k$ is sought.
The correction $s_k$ is chosen to improve a model $m_k(s)$ of
the objective function $f(x_k+s)$ built around
$x_k$. The model is the sum of two basic components,
a suitable approximation $t_k(s)$ of $f(x_k+s)$,
%another approximation of $(\rho/r) \|x_k+s\|_r^r$ (if $\rho > 0$),
and a regularization term $(\sigma_k/p) \|s\|_{S_k}^p$
involving a weight $\sigma_k$, power $p$ and
a norm $\|s\|_{S_k} := \sqrt{s^T S_k s}$ for a given positive
definite scaling matrix $S_k$ that is included to prevent large
corrections. The weight  $\sigma_k$ is adjusted as the algorithm
progresses to  ensure convergence.

The model $t_k(s)$ is a truncated Taylor-series approximation, and this
relies on being able to compute or estimate derivatives of $c(x)$.
Various models are provided, and each has different derivative requirements.
We denote the $m$ by $n$ residual **Jacobian**
$J(x) \equiv \nabla_x c(x)$ as the matrix  whose $i,j$-th component
$$J(x)_{i,j} := \partial c_i(x) / \partial x_j \;\;
\mbox{for $i=1,\ldots,m$ and $j=1,\ldots,n$.}$$
For a given $m$-vector $y$, the
**weighted-residual Hessian** is the sum
$$H(x,y) := \sum_{\ell=1}^m y_{\ell} H_{\ell}(x), \;\; \mbox{where}\;\; H_{\ell}(x)_{i,j} := \partial^2 c_{\ell}(x) / \partial x_i \partial x_j \;\; \mbox{for $i,j=1,\ldots,n$}$$
is the Hessian of $c_\ell(x)$.
Finally, for a given vector $v$, we define
the **residual-Hessians-vector product** matrix
$$P(x,v) := (H_1(x) v, \ldots, H_m(x) v).$$
The models $t_k(s)$ provided are,

1. the **first-order Taylor** approximation
   $f(x_k) + g(x_k)^T s$, where $g(x) = J^T(x) W c(x)$,

2. a **barely second-order** approximation
   $f(x_k) + g(x_k)^T s + \frac{1}{2} s^T W s$,

3. the **Gauss-Newton** approximation
   $\frac{1}{2} \| c(x_k) + J(x_k) s\|^2_W$,

4. the **Newton (second-order Taylor)** approximation

   $f(x_k) + g(x_k)^T s + \frac{1}{2} s^T [ J^T(x_k) W J(x_k) + H(x_k,W c(x_k))] s$, and

5. the **tensor Gauss-Newton** approximation
   $\frac{1}{2} \| c(x_k) + J(x_k) s + \frac{1}{2} s^T \cdot P(x_k,s) \|^2_W$,
   where the $i$-th component of $s^T \cdot P(x_k,s)$ is
   shorthand for the scalar $s^T H_i(x_k) s$,
   where $W$ is the diagonal matrix of weights
   $w_i$, $i = 1, \ldots m$0.

method
------

An adaptive regularization method is used.
In this, an improvement to a current
estimate of the required minimizer, $x_k$ is sought by computing a
step $s_k$. The step is chosen to approximately minimize a model $t_k(s)$
of $f_{\rho,r}(x_k+s)$
that includes a weighted regularization term
$\frac{\sigma_k}{p} \|s\|_{S_k}^p$
for some specified positive weight $\sigma_k$. The quality of the
resulting step $s_k$ is assessed by computing the "ratio"
$(f(x_k) - f(x_k + s_k))/(t_k(0) - t_k(s_k))$.
The step is deemed to have succeeded if the ratio exceeds a given $\eta_s > 0$,
and in this case $x_{k+1} = x_k + s_k$. Otherwise
$x_{k+1} = x_k$, and the weight is increased by powers of a given
increase factor up to a given limit. If the ratio is larger than
$\eta_v \geq \eta_d$, the weight will be decreased by powers of a given
decrease factor again up to a given limit. The method will terminate
as soon as $f(x_k)$ or
$\|\nabla_x f(x_k)\|$ is smaller than a specified value.

A choice of linear, quadratic or quartic models $t_k(s)$ is available
(see the previous section), and normally a two-norm regularization will
be used, but this may change if preconditioning is employed.

If linear or quadratic models are employed, an appropriate,
approximate model minimizer is found using either a direct approach
involving factorization of a shift of the model Hessian $B_k$ or an
iterative (conjugate-gradient/Lanczos) approach based on approximations
to the required solution from a so-called Krlov subspace. The direct
approach is based on the knowledge that the required solution
satisfies the linear system of equations $(B_k + \lambda_k I) s_k
= - \nabla_x f(x_k)$ involving a scalar Lagrange multiplier $\lambda_k$.
This multiplier is found by uni-variate root finding, using a safeguarded
Newton-like process, by ``RQS``. The iterative approach
uses ``GLRT``, and is best accelerated by preconditioning with
good approximations to the Hessian of the model using ``PSLS``. The
iterative approach has the advantage that only Hessian matrix-vector products
are required, and thus the Hessian $B_k$ is not required explicitly.
However when factorizations of the Hessian are possible, the direct approach
is often more efficient.

When a quartic model is used, the model is itself of least-squares form,
and the package calls itself recursively to approximately minimize its
model. The quartic model often gives a better approximation, but at the
cost of more involved derivative requirements.

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
  **53(1)** (2012) 1--22.

The specific methods employed here are discussed in

  N. I. M. Gould, J. A. Scott and T. Rees,
  ``Convergence and evaluation-complexity analysis of a regularized
  tensor-Newton method for solving nonlinear least-squares problems''.
  *Computational Optimization and Applications*
  **73(1)** (2019) 1--35.
