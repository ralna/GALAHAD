purpose
-------

The ``dps`` package **constructs a symmetric, positive definite matrix** $M$ 
from a given $H$ so that $H$ is is **diagonal** 
in the norm $\|v\|_M = \sqrt{v^T M v}$ induced by $M$, and consequently 
minimizers of **trust-region** and **regularized quadratic** subproblems 
may be computed efficiently.
The aim is either to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
where the vector $x$ is required to satisfy 
the ellipsoidal  **trust-region constraint** $\|x\|_{M} \leq  \Delta$, 
or to minimize the regularized quadratic objective
$$r(x) = q(x) + \frac{\sigma}{p} \|x\|_M^p,$$
where the **radius** $\Delta > 0$, the **weight** $\sigma > 0$, 
and the **power** $p \geq 2$.
A factorization of the matrix $H$ will be required, so this package is
most suited for the case where such a factorization,
either dense or sparse, may be found efficiently.

See Section 4 of $GALAHAD/doc/dps.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$H x_* + \lambda_* M x_* + g = 0$,
where $\lambda_* \geq 0$ is a Lagrange
multiplier that corresponds to the constraint
$\|x\|_M  \leq  \Delta$ in the trust-region case,
and is given by $\lambda_* = \sigma \|x_*\|^{p-2}$
for the regularization problem involve $r(x)$.
In addition $H + \lambda_* M$ will be positive semi-definite; in
most instances it will actually be positive definite, but in special
"hard" cases singularity is a possibility.

The matrix $H$ is decomposed as
$$H = P L D L^T P^T$$
by calling the matrix factorization package ``SLS``. Here $P$ is a permutation
matrix, $L$ is unit lower triangular and $D$ is block diagonal, with
blocks of dimension at most two. The spectral decomposition of each diagonal
block of $D$ is computed, and each eigenvalue $\theta$ is replaced by
$\max ( | \theta | , \theta_{\min} ) $,
where $\theta_{\min}$ is a positive user-supplied value. The resulting block
diagonal matrix is $B$, from which we define the **modified-absolute-value**
$$M = P L B L^T P^T;$$
an alternative due to Goldfarb uses instead the simpler
$$M = P L L^T P^T.$$

Given the factors of $H$ (and $M$), the required solution is
found by making the change of variables $y = B^{1/2} L^T P^T x$
(or $y = L^T P^T x$ in the Goldfarb case)
which results in ``diagonal'' trust-region and regularization subproblems,
whose solution may be easily obtained suing a Newton or higher-order iteration
of a resulting "secular" equation. If subsequent problems, for which
$H$ and $g$ are unchanged, are to be attempted, the existing
factorization and solution may easily be exploited.

The dominant cost is that for the factorization of the symmetric, but
potentially indefinite, matrix $H$ using the package ``SLS``.

references
----------

The method is described in detail for the trust-region case in

  N. I. M. Gould and J. Nocedal,
  ``The modified absolute-value factorization for trust-region minimization''.
  In ``High Performance Algorithms and Software in Nonlinear Optimization''
  (R. De Leone, A. Murli, P. M. Pardalos and G. Toraldo, eds.),
  Kluwer Academic Publishers, (1998) 225--241,

while the adaptation for the regularization case is obvious. The method used
to solve the diagonal trust-region and regularization subproblems are as
given by

  H. S. Dollar, N. I. M. Gould and D. P. Robinson,
  ``On solving trust-region and other regularised subproblems in optimization''.
  *Mathematical Programming Computation* **2(1)** (2010) 21--57

with simplifications due to the diagonal Hessian.
