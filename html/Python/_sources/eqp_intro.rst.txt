purpose
-------

The ``eqp`` package uses an **iterative method** to solve a
given **equality-constrained quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
or the **shifted-least-distance** objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear equality constraints
$$A x + c = 0,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric and $m$ by $n$ general matrices,  
$g$, $w$, $x^0$ and $c$ are vectors, and  $f$ is a scalar.
The method is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/eqp.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x + c = 0 \;\;\mbox{(1)}$$
and the **dual optimality conditions**
$$H x + g = A^T y,$$
where the vector $y$ is known as the **Lagrange multipliers** for
the general linear constraints.

In the shifted-least-distance case, $g$ is shifted by $-W^2 x^0$,
and $H = W^2$, where $W$ is the diagonal matrix whose entries are the $w_j$.

method
------

A solution to the problem is found in two phases.
In the first, a point $x_F$ satisfying (1) is found.
In the second, the required solution $x = x_F + s$
is determined by finding $s$ to minimize 
$q(s) = \frac{1}{2} s^T H s + g_F^T s + f_F^{}$
subject to the homogeneous constraints $A s = 0$,
where $g_F^{} = H x_F^{} + g$ and 
$f_F^{} = \frac{1}{2} x_F^T H x_F^{} + g^T x_F^{} + f$.
The required constrained minimizer of $q(s)$ is obtained
by implictly applying the preconditioned conjugate-gradient method
in the null space of $A$. Any preconditioner of the form
$$K_{G} = \left(\begin{array}{cc} G & A^T \\ A  & 0 \end{array}\right)$$
is suitable, and the package ``SBLS``
provides a number of possibilities. In order to ensure that the
minimizer obtained is finite, an additional, precautionary trust-region
constraint $\|s\| \leq \Delta$ for some suitable positive radius 
$\Delta$ is imposed, and the package ``GLTR`` is used to solve 
this additionally-constrained problem.

references
----------

The preconditioning aspcets are described in detail in

  H. S. Dollar, N. I. M. Gould and A. J. Wathen.
  ``On implicit-factorization constraint preconditioners''.
  In  Large Scale Nonlinear Optimization (G. Di Pillo and M. Roma, eds.)
  Springer Series on Nonconvex Optimization and Its Applications, Vol. 83,
  Springer Verlag (2006) 61--82

and

  H. S. Dollar, N. I. M. Gould, W. H. A. Schilders and A. J. Wathen
  ``On iterative methods and implicit-factorization preconditioners for 
  regularized saddle-point systems''.
  *SIAM Journal on Matrix Analysis and Applications* 
  **28(1)** (2006) 170--189,

while the constrained conjugate-gradient method is discussed in

  N. I. M. Gould, S. Lucidi, M. Roma and Ph. L. Toint, 
  ``Solving the trust-region subproblem using the Lanczos method''. 
  *SIAM Journal on Optimization* **9(2)** (1999), 504--525. 
