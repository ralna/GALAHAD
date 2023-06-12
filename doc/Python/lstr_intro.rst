purpose
-------

The ``lstr`` package uses a **Krylov-subspace iteration** to find an 
approximation of the global minimizer of the 
**linear sum-of-squares objective function** 
within a **sphere**; this is commonly known as the
**linear least-squares trust-region subproblem**.
The aim is to minimize the least-squares objective function
$$q(x) = \frac{1}{2}\|Ax - b\|_2^2,$$ 
where the vector $x$ is required to satisfy 
the spherical **trust-region constraint** $\|x\|_2 \leq  \Delta$, 
where the $\ell_2$-norm of $x$ is defined to be $\|x\|_2 = \sqrt{x^T x}$,
and where the **radius** $\Delta > 0$.
The method may be suitable for large problems as no factorization is
required. Reverse communication is used to obtain
matrix-vector products of the form $u + A v$ and $v + A^T u.$

See Section 4 of $GALAHAD/doc/lstr.pdf for additional details.

method
------

The required solution $x$ necessarily satisfies the optimality condition
$A^T ( A x - b ) + \lambda x = 0$, where $\lambda \geq 0$
is a Lagrange multiplier corresponding to the trust-region constraint
$\|x\|_2  \leq  \Delta$.

The method is iterative. Starting  with the vector $u_1 = b$, a
bi-diagonalisation process is used to generate the vectors $v_k$ and
$u_k+1$ so that the $n$ by $k$ matrix $V_k = ( v_1 \ldots v_k)$
and the $m$ by $(k+1)$ matrix $U_k = ( u_1 \ldots u_{k+1})$
together satisfy
$$A V_k = U_{k+1} B_k \;\;\mbox{and}\;\;  b = \|b\| U_{k+1} e_1,$$
where $B_k$ is $(k+1)$ by $k$ and lower bi-diagonal, $U_k$ and
$V_k$ have orthonormal columns and $e_1$ is the first unit vector.
The solution sought is of the form $x_k = V_k y_k$, where $y_k$
solves the bi-diagonal least-squares trust-region problem
$$\min \| B_k y - \|b\| e_1 \|_2 \;\;\mbox{subject to}\;\;
\|y\|_2 \leq \Delta.\;\;\mbox{(1)}$$

If the trust-region constraint is inactive, the solution $y_k$
may be found, albeit indirectly, via the LSQR algorithm of Paige and Saunders
which solves the bi-diagonal least-squares problem
$$\min \| B_k y - \|b\| e_1 \|_2$$
using a QR factorization of $B_k$. Only the most recent $v_k$ and $u_{k+1}$
are required, and their predecessors discarded, to compute $x_k$ from
$x_{k-1}$. This method has the important property that the iterates
$y$ (and thus $x_k$) generated increase in norm with $k$. Thus
as soon as an LSQR iterate lies outside the trust-region, the required solution
to (1) and thus to the original problem must lie on the boundary of the
trust-region.

If the solution is so constrained, the simplest strategy is to interpolate
the last interior iterate with the newly discovered exterior one to find the
boundary point --- the so-called Steihaug-Toint point --- between them.
Once the solution is known to lie on the trust-region boundary,
further improvement may be made by solving
$$\min \| B_k y - \|b\| e_1 \|_2
\;\;\mbox{subject to}\;\; \|y\|_2 = \Delta,$$
for which the optimality conditions require that $y_k = y(\lambda_k)$
where $\lambda_k$ is the positive root of
$$B_k^T ( B_k^{} y(\lambda) - \|b\| e_1^{} ) + \lambda
y(\lambda) = 0 \;\;\mbox{and}\;\;  \|y(\lambda)\|_2 = \Delta$$
The vector $y(\lambda)$ is equivalently the solution to the
regularized least-squares problem
$$\min \left \| \left(\begin{array}{c}
B_k \\ \lambda^{1/2} I
\end{array}\right) y - \|b\| e_1^{} \right \|_2$$
and may be found efficiently. Given  $y(\lambda)$, Newton's method
is then used to find $\lambda_k$ as the positive root of
$\|y(\lambda)\|_2 = \Delta$. Unfortunately, unlike when the solution
lies in the interior of the trust-region, it is not known how to recur
$x_k$ from $x_{k-1}$ given $y_k$, and a second pass in which
$x_k = V_k y_k$ is regenerated is needed --- this need only be done
once $x_k$ has implicitly deemed to be sufficiently close to optimality.
As this second pass is an additional expense, a record is kept of the
optimal objective function values for each value of $k$, and the second
pass is only performed so far as to ensure a given fraction of the
final optimal objective value. Large savings may be made in the second
pass by choosing the required fraction to be significantly smaller than one.

references
----------

A complete description of the unconstrained case is given by

  C. C. Paige and M. A. Saunders,
  ``LSQR: an algorithm for sparse linear equations and sparse least  squares''.
  *ACM Transactions on Mathematical Software* **8(1** (1982) 43--71,

and

  C. C. Paige and M. A. Saunders,
  ``ALGORITHM 583: LSQR: an algorithm for sparse linear equations and
  sparse least squares''.
  *ACM Transactions on Mathematical Software* **8(2)** (1982) 195--209.

Additional details on how to proceed once the trust-region constraint is
encountered are described in detail in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  ``Trust-region and other regularisation of linear
  least-squares problems''.
  *BIT* **49(1)** (2009) 21-53.
