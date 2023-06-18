purpose
-------

The ``glrt`` package uses a **Krylov-subspace iteration** to find an 
approximation of the global minimizer of 
**regularized quadratic objective function**. 
The aim is to minimize the regularized quadratic objective function
$$r(x) = f + g^T x + \frac{1}{2} x^T H x + \frac{\sigma}{p} \|x\|_{M}^p,$$ 
where the **weight** $\sigma \geq 0$, the **power** $p \geq 2$, 
and where the $M$-norm of $x$ is defined to be $\|x\|_{M} = \sqrt{x^T M x}$.
The method may be suitable for large problems as no factorization of $H$ is
required. Reverse communication is used to obtain
matrix-vector products of the form $H z$ and $M^{-1} z.$

See Section 4 of $GALAHAD/doc/glrt.pdf for additional details.

method
------

The required solution $x$ necessarily satisfies the optimality condition
$H x + \lambda M x + g = 0$, where
$\lambda = \sigma \|x\|_{M}^{p-2}$.
In addition, the matrix $H + \lambda M$ will be positive semi-definite.

The method is iterative. Starting  with the vector $M^{-1} g$,
a matrix of Lanczos vectors is built one column at a time
so that the $k$-th column is generated during
iteration $k$. These columns span a so-called Krylov space.
The resulting $n$ by $k$ matrix $Q_k $ has the
property that $Q_{k}^T H Q_k^{}  =  T_{k}^{}$,
where $T_k$ is tridiagonal. An approximation to the
required solution may then be expressed formally as
$$x_{k+1}  =  Q_k y_k$$
where $y_k $ solves the ``tridiagonal'' subproblem of minimizing
$$\frac{1}{2} y^T T_k y  + \|g\|_{M^{-1} } e_{1}^T y +
\frac{1}{p} \sigma \| y \|_2^p,\;\;\mbox{(1)}$$
where $e_1$ is the first unit vector.

To minimize (1), the optimality conditions
$$( T_k + \lambda I ) y(\lambda) = - g,\;\;\mbox{(2)}$$
where $\lambda = \sigma \|y(\lambda)\|_{M}^{p-2}$
are used as the basis of an iteration. Specifically, given an estimate
$\lambda$ for which $T_k + \lambda I$ is positive definite,
the tridiagonal system (2) may be efficiently solved to give
$y(\lambda)$. It is then simply a matter of adjusting $\lambda$
(for example by a Newton-like process) to solve the scalar nonlinear equation
$$\theta(\lambda) \equiv \|y(\lambda)\|_{M}^{p-2}
- \frac{\lambda}{\sigma} = 0.\;\;\mbox{(3)}$$
In practice (3) is reformulated, and 
a more rapidly converging iteration is used.

It is possible to measure the optimality measure
$\|H x + \lambda M x + g\|_{M^{-1}}$
without computing $x_{k+1}$, and thus without
needing $Q_k $. Once this measure is sufficiently small, a second pass
is required to obtain the estimate $x_{k+1} $ from $y_k $.
As this second pass is an additional expense, a record is kept of the
optimal objective function values for each value of $k$, and the second
pass is only performed so far as to ensure a given fraction of the
final optimal objective value. Large savings may be made in the second
pass by choosing the required fraction to be significantly smaller than one.

Special code is used in the special case $p=2$, as in this case a single
pass suffices.

reference
---------

The method is described in detail in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  ``Adaptive cubic regularisation methods for unconstrained
  optimization. Part {I}: motivation, convergence and numerical results''.
  *Mathematical Programming* **127(2)** (2011) 245-295.
