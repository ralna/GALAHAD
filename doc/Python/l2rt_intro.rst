purpose
-------

The ``l2rt`` package uses a **Krylov-subspace iteration** to find an 
approximation of the global minimizer of the 
**regularized linear Euclidean-norm objective function**. 
The aim is to minimize the objective function
$$r(x) = \sqrt{\|Ax - b\|_2^2 + \mu \|x\|_2^2} + \frac{\sigma}{p} \|x\|_2^p,$$ 
where the $\ell_2$-norm of $x$ is defined to be $\|x\|_2 = \sqrt{x^T x}$,
and where the **shift** $\mu \geq 0$, **weight** $\sigma > 0$ and 
**power** $p \geq 2$.
The method may be suitable for large problems as no factorization is
required. Reverse communication is used to obtain
matrix-vector products of the form $u + A v$ and $v + A^T u.$

See Section 4 of $GALAHAD/doc/l2rt.pdf for additional details.

method
------

The required solution $x$ necessarily satisfies the optimality condition
$A^T ( A x - b ) + \lambda x = 0$, where
$\lambda = \mu + \sigma \|x\|_2^{p-2}
\sqrt{\| A x - b \|_2^2 + \mu \|x\|_2^2}.$

The method is iterative. Starting  with the vector $u_1 = b$, a
bi-diagonalisation process is used to generate the vectors $v_k$ and
$u_k+1$ so that the $n$ by $k$ matrix $V_k = ( v_1 \ldots v_k)$
and the $m$ by $(k+1)$ matrix $U_k = ( u_1 \ldots u_{k+1})$
together satisfy
$$A V_k = U_{k+1} B_k \;\;\mbox{and}\;\; b = \|b\|_2 U_{k+1} e_1,$$
where $B_k$ is $(k+1)$ by $k$ and lower bi-diagonal, $U_k$ and
$V_k$ have orthonormal columns and $e_1$ is the first unit vector.
The solution sought is of the form $x_k = V_k y_k$, where $y_k$
solves the bi-diagonal regularized Euclidean-norm problem
$$\min \sqrt{\| B_k y - \|b\| e_1 \|_2^2 + \mu \|y\|_2^2}
+ \frac{1}{p} \sigma \|  y \|_2^p.\;\;\mbox{(1)}$$

To solve (1), the optimality conditions
$$( B_k^T ( B_k^{} y(\lambda) - \|b\| e_1^{} ) + \lambda y(\lambda) = 0,$$
where
$$\lambda = \mu + \sigma \|y(\lambda)\|_2^{p-2}
\sqrt{ \| B_k y(\lambda) - \|b\| e_1 \|_2^2
+ \mu \|y(\lambda)\|_2^2},$$
are used as the basis of an iteration.
The vector $y(\lambda)$ is equivalently the solution to the
regularized least-squares problem
$$\min \left \| \left(\begin{array}{c}
B_k \\ \lambda^{1/2} I
\end{array}\right) y - \|b\| e_1^{} \right \|_2.$$
Thus, given an estimate $\lambda \geq 0$, this regularized 
least-squares problem may be efficiently solved to give $y(\lambda)$.
It is then simply a matter of adjusting $\lambda$
(for example by a Newton-like process) to solve the scalar nonlinear equation
$$\theta(\lambda) \equiv
\sigma \|y(\lambda)\|_2^{p-2} \sqrt{ \| B_k y(\lambda) - \|b\| e_1 \|_2^2 +
\mu \|y(\lambda)\|_2^2} + \mu - \lambda = 0.\;\;\mbox{(2)}$$
In practice (2) is reformulated, and a more rapidly converging 
iteration is used. Having found  $y_k$, a second pass in which
$x_k = V_k y_k$ is regenerated is needed---this need only be done
once $x_k$ has implicitly deemed to be sufficiently close to optimality.
As this second pass is an additional expense, a record is kept of the
optimal objective function values for each value of $k$, and the second
pass is only performed so far as to ensure a given fraction of the
final optimal objective value. Large savings may be made in the second
pass by choosing the required fraction to be significantly smaller than one.

Special code is used in the special case $p=2$ as in this case the
equation (2) significantly simplifies.

references
----------

A complete description of the un- an quadratically-regularized 
cases is given by

  C. C. Paige and M. A. Saunders,
  ``LSQR: an algorithm for sparse linear equations and sparse least  squares''.
  *ACM Transactions on Mathematical Software* **8(1** (1982) 43--71,

and

  C. C. Paige and M. A. Saunders,
  ``ALGORITHM 583: LSQR: an algorithm for sparse linear equations and
  sparse least squares''.
  *ACM Transactions on Mathematical Software* **8(2)** (1982) 195--209.

Additional details on the Newton-like process needed to determine $\lambda$ and
other details are described in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  ``Trust-region and other regularisation of linear
  least-squares problems''.
  *BIT* **49(1)** (2009) 21-53.

