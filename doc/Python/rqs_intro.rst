purpose
-------

The ``rqs`` package uses **matrix factorization** to find the 
global minimizer of a **regularized quadratic objective function**.
The aim is to minimize the regularized quadratic objective function
$$r(x) = f + g^T x + \frac{1}{2} x^T H x + \frac{\sigma}{p} \|x\|_{M}^p,$$ 
where the **weight** $\sigma > 0$, the **power** $p \geq 2$, the  vector $x$
may optionally  be required to satisfy **affine constraints** $A x = 0,$
and where the $M$-norm of $x$ is defined to be $\|x\|_{M} = \sqrt{x^T M x}$.

The matrix $M$ need not be provided in the commonly-occurring
$\ell_2$-regularization case for which $M = I$, the $n$ by $n$
identity matrix.

Factorization of matrices of the form $H + \lambda M$, or
$$\left(\begin{array}{cc} H + \lambda M & A^T \\ A & 0 \end{array}\right)$$
in cases where $A x = 0$ is imposed, for a succession
of scalars $\lambda$ will be required, so this package is most suited
for the case where such a factorization may be found efficiently. If
this is not the case, the package ``glrt`` may be preferred.

See Section 4 of $GALAHAD/doc/rqs.pdf for a brief description of the
method employed and other details.

See Section 4 of $GALAHAD/doc/trs.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$H x_* + \lambda_* M x_* + A^T y_* + g = 0$ and
$A x_* = 0$, where $\lambda_* = \sigma \|x_*\|^{p-2}$
is a Lagrange multiplier corresponding to the regularization, and
$y_*$ are Lagrange multipliers for the linear constraints $A x = 0$, if any.
In addition in all cases, the matrix $H + \lambda_* M$ will be positive
semi-definite on the null-space of $A$; in most instances it will actually
be positive definite, but in special "hard" cases singularity is a 
possibility.

The method is iterative, and proceeds in two phases.
Firstly, lower and upper bounds, $\lambda_L$ and
$\lambda_U$, on $\lambda_*$ are computed
using Gershgorin's theorems and other eigenvalue bounds. The
first phase of the computation proceeds by progressively shrinking  the bound
interval $[\lambda_L,\lambda_U]$
until a value $\lambda$ for which
$\|x(\lambda)\|_{M}  \geq  \sigma \|x(\lambda)\|_M^{p-2}$ is found.
Here $x(\lambda)$ and its companion $y(\lambda)$
are defined to be a solution of
$$(H + \lambda M)x(\lambda) + A^T y(\lambda) = 
- g \;\;\mbox{and}\;\; A x(\lambda) = 0.\;\;\mbox{(2)}$$
Once the terminating $\lambda$ from the first phase has been discovered,
the second phase consists of applying Newton or higher-order iterations 
to the nonlinear "secular" equation 
$\lambda = \sigma \|x(\lambda)\|_M^{p-2}$ with
the knowledge that such iterations are both globally and ultimately
rapidly convergent. It is possible in the "hard" case that the
interval in the first-phase will shrink to the single point $\lambda_*$,
and precautions are taken, using inverse iteration with Rayleigh-quotient
acceleration to ensure that this too happens rapidly.

The dominant cost is the requirement that we solve a sequence of 
linear systems (2). In the absence of linear constraints, an
efficient sparse Cholesky factorization with precautions to
detect indefinite $H + \lambda M$ is used. If $A x  = 0$ is required,
a sparse symmetric, indefinite factorization of (1) is used
rather than a Cholesky factorization.

reference
---------
The method is described in detail in

  H. S. Dollar, N. I. M. Gould and D. P. Robinson.
  ``On solving trust-region and other regularised subproblems in optimization''.
  *Mathematical Programming Computation* **2(1)** (2010) 21--57.
