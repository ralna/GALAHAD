purpose
-------

Given a real $m$ by $n$ model matrix $A$, a real $n$ by $n$ symmetric 
diagonally-dominant matrix $S$, a real $m$ vector of observations $b$ 
and a scalar $\Delta>0$, the ``llst`` package finds a **minimizer of the linear 
least-squares objective function $\|A x - b\|_2$, where the vector $x$ is
required to satisfy the constraint $\|x\|_S \leq  \Delta$**,
and where the $S$-norm of $x$ is $\|x\|_S = \sqrt{x^T S x}$.
This problem commonly occurs as a trust-region subproblem in nonlinear
least-squares calculations.
The package may also be used to solve the related problem in which $x$ is
instead required to satisfy the
**equality constraint $\|x\|_S = \Delta$**.
The matrix $S$ need not be provided in the commonly-occurring
$\ell_2$-trust-region case for which $S = I$, the $n$ by $n$
identity matrix.

Factorization of matrices of the form
$$\begin{pmatrix}\lambda S & A^T \\ A & - I\end{pmatrix}\mspace{5em}\mbox{(1)}$$
for a succession of scalars $\lambda$ will be required, so this package is 
most suited for the case where such a factorization may be found efficiently. 
If this is not the case, the package ``lstr`` may be preferred.

See Section 4 of $GALAHAD/doc/llst.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$A^T A x_* + \lambda_* S x_* = A^T b$, where $\lambda_* \geq 0$ is a Lagrange
multiplier corresponding to the constraint $\|x\|_S  \leq  \Delta$;
for the equality-constrained problem $\|x\|_S = \Delta$ and
the multiplier is unconstrained.

The method is iterative, and proceeds in two phases.
Firstly, lower and upper bounds, $\lambda_L$ and
$\lambda_U$, on $\lambda_*$ are computed
using Gershgorin's theorems and other eigenvalue bounds,
including those that may involve the Cholesky factorization of $S$  The
first phase of the computation proceeds by progressively shrinking  the bound
interval $[\lambda_L,\lambda_U]$
until a value $\lambda$ for which
$\|x(\lambda)\|_S  \geq  \Delta$ is found.
Here $x(\lambda)$ and its companion $y(\lambda)$
are defined to be a solution of
$$(A^T A  + \lambda S)x(\lambda) = A^T b; \mspace{5em}\mbox{(2)}$$
along the way the possibility that
$\|x(0)\|_S  \leq  \Delta$ is examined, and if this transpires
the process is terminated with $x_* = x(0)$. Once the terminating 
$\lambda$ from the first phase has been discovered, the second phase
consists of applying Newton or higher-order iterations to the nonlinear
*secular* equation $\|x(\lambda)\|_S = \Delta$ with
the knowledge that such iterations are both globally and ultimately
rapidly convergent.

The dominant cost is the requirement that we solve a sequence of 
linear systems (2). This may be rewritten as
$$\begin{pmatrix}\lambda S & A^T \\ A & - I\end{pmatrix}
\begin{pmatrix}x(\lambda) \\ y(\lambda)\end{pmatrix} =
\begin{pmatrix}A^T b \\ 0\end{pmatrix} \mspace{5em} \mbox{(3)}$$
for some auxiliary vector $y(\lambda)$.
In general a sparse symmetric, indefinite factorization of the
coefficient matrix of (3) is
often preferred to a Cholesky factorization of that of (2).

reference
---------

The method is the obvious adaptation to the linear least-squares
problem of that described in detail in

  H. S. Dollar, N. I. M. Gould and D. P. Robinson.
  ``On solving trust-region and other regularised subproblems in optimization''.
  *Mathematical Programming Computation* **2(1)** (2010) 21--57.
