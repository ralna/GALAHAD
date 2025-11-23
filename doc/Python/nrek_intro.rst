purpose
-------

The ``nrek`` package uses an **extended-Krylov-subspace iteration** to find the 
global minimizer of a **norm-regularized quadratic objective function**;
this is commonly known as the **norm-regularization subproblem**.
The aim is to minimize the regularized quadratic objective function
$$r(x) = f + c^T x + \frac{1}{2} x^T H x + \frac{1}{p} \sigma \|x\|_S^p,$$ 
where the $S$-norm of $x$ is defined to be $\|x\|_S = \sqrt{x^T S x}$,
and where the **weight** $\sigma > 0$ and **power** $p>2$.
The matrix $S$ need not be provided in the commonly-occurring
$\ell_2$-trust-region case for which $S = I$, the $n$ by $n$
identity matrix.

Factorization of the matrices $H$ and, if present, $S$
will be required, so this package is most suited
for the case where such a factorization may be found efficiently. If
this is not the case, the package ``glrt`` may be preferred.

See Section 4 of $GALAHAD/doc/nrek.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$H x_* + \lambda_* S x_* + c = 0$, where $\lambda_* \sigma \|x_*\|^{p-2}$
is a Lagrange multiplier for the regularization.
In addition in all cases, the matrix $H + \lambda_* S$ will be positive
semi-definite; in most instances it will actually be positive definite, 
but in special "hard" cases singularity is a possibility.

The method is iterative, and is based upon building a solution approximation
from an orthogonal basis of the evolving extended Krylov subspaces
${\cal K}_{2m+1}(H,c) = \mbox{span}\{c,H^{-1}c,H c,H^{-2}c,H^2c,\ldots,$
$H^{-m}c,H^{m}c\}$
as $m$ increases. The key observations are (i) the manifold of
solutions to the optimality system
\[ ( H + \lambda I ) x(\lambda) = - c\]
as a function of $\sigma$ is of approximately very low rank, (ii)
the subspace ${\cal K}_{2m+1}(H,c)$ rapidly gives a very good 
approximation to this manifold, (iii) it is straightforward to
build an orthogonal basis of ${\cal K}_{2m+1}(H,c)$
using short-term recurrences and a single factorization of $H$, and
(iv) solutions to the norm-regularization subproblem restricted to elements
of the orthogonal subspace may be found very efficiently 
using effective high-order root-finding methods. 
Coping with general scalings $S$ is a straightforward extension so long
as factorization of $S$ is also possible.

reference
---------

The method is described in detail in

  H. Al Daas and N. I. M. Gould.
  Extended-Krylov-subspace methods for trust-region and norm-regularization 
  subproblems. Preprint STFC-P-2025-002, Rutherford Appleton Laboratory,
  Oxfordshire, England.
