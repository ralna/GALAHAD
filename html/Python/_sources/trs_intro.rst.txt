purpose
-------

The ``trs`` package uses **matrix factorization** to find the 
global minimizer of a **quadratic objective function** within
an **ellipsoidal region**; this is commonly known as the
**trust-region subproblem**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
where the vector $x$ is required to satisfy 
the ellipsoidal  **trust-region constraint** $\|x\|_M \leq  \Delta$, 
and optionally  **affine constraints** $A x = 0$, 
where the $M$-norm of $x$ is defined to be $\|x\|_M = \sqrt{x^T M x}$,
and where the **radius** $\Delta > 0$.

The package may also be used to solve the related problem in which $x$ is
instead required to satisfy the **equality constraint** $\|x\|_M = \Delta$.
The matrix $M$ need not be provided in the commonly-occurring
$\ell_2$-trust-region case for which $M = I$, the $n$ by $n$
identity matrix.

Factorization of matrices of the form $H + \lambda M$, or
$$\left(\begin{array}{cc} 
H + \lambda M & A^T \\ A & 0 
\end{array}\right)\;\;\mbox{(1)}$$
in cases where $A x = 0$ is imposed, for a succession
of scalars $\lambda$ will be required, so this package is most suited
for the case where such a factorization may be found efficiently. If
this is not the case, the package ``gltr`` may be preferred.

See Section 4 of $GALAHAD/doc/trs.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$H x_* + \lambda_* M x_* + A^T y_* + g = 0$ and
$A x_* = 0$, where $\lambda_* \geq 0$ is a Lagrange
multiplier corresponding to the constraint $\|x\|_M  \leq  \Delta$
and $y_*$ are Lagrange multipliers for the linear constraints $A x = 0$,
if any;
for the equality-constrained problem $\|x\|_M = \Delta$,
the multiplier is unconstrained.
In addition in all cases, the matrix $H + \lambda_* M$ will be positive
semi-definite on the null-space of $A$; in most instances it will actually
be positive definite, but in special ``hard'' cases singularity is a 
possibility.

The method is iterative, and proceeds in two phases.
Firstly, lower and upper bounds, $\lambda_L$ and
$\lambda_U$, on $\lambda_*$ are computed
using Gershgorin's theorems and other eigenvalue bounds. The
first phase of the computation proceeds by progressively shrinking  the bound
interval $[\lambda_L,\lambda_U]$
until a value $\lambda$ for which
$\|x(\lambda)\|_M  \geq  \Delta$ is found.
Here $x(\lambda)$ and its companion $y(\lambda)$
are defined to be a solution of
$$(H + \lambda M)x(\lambda) + A^T y(\lambda) = 
- g \;\;\mbox{and}\;\; A x(\lambda) = 0;\;\;\mbox{(2)}$$
along the way the possibility that $H$ might be positive definite on
the null-space of $A$ and
$\|x(0)\|_M  \leq  \Delta$ is examined, and if this transpires
the process is terminated with $x_* = x(0)$.
Once the terminating $\lambda$ from the first phase has been discovered,
the second phase
consists of applying Newton or higher-order iterations to the nonlinear
"secular" equation $\|x(\lambda)\|_M  =  \Delta$ with
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
