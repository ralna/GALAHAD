purpose
-------

The ``dqp`` package uses a 
**dual gradient-projection method** to solve a given
**stricly-convex quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$
or the **shifted-least-distance**  objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric postive-definite and $m$ by $n$ matrices,  
$g$, $w$ and $x^0$ are vectors, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/dqp.pdf for additional details.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$H x + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; z = z_l + z_u,$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

In the shifted-least-distance case, $g$ is shifted by $-W^2 x^0$,
and $H = W^2$, where $W$ is the diagonal matrix whose entries are the $w_j$.

method
------
Dual gradient-projection methods solve the quadratic programmimg problem
by instead solving the dual quadratic program
$$\begin{array}{ll}\mbox{minimize}\;\;
q^D(y^{l}, y^{u}, z^{l}, z^{u}) = & \!\!\!
\frac{1}{2} [ ( y^{l0}  + y^{u} )^T A + ( z^{l} + z^{u} ]^T ) H^{-1}
[ A^T ( y^{l}  + y^{u} ) + z^{l} + z^{u} ] \\
& - [ ( y^{l}  + y^{u} )^T A + ( z^{l} + z^{u} ]^T ) H^{-1} g
- ( c^{l T} y^{l} + c^{u T} y^{u} +
x^{l T} z^{l} + x^{u T} z^{u}) \\
\mbox{subject to} & ( y^{l}, z^{l} ) \geq 0 \;\;\mbox{and} \;\;
(y^{u}, z^{u}) \leq 0,\end{array}$$
and then recovering the required solution from the linear system
$$H x = - g + A^T ( y^{l}  + y^{u} ) + z^{l} + z^{u}.$$
The dual problem is solved by an accelerated gradient-projection
method comprising of alternating phases in which (i) the current
projected dual gradient is traced downhill (the 'arc search')
as far as possible and (ii) the  dual variables that
are currently on their bounds are temporarily fixed and the unconstrained
minimizer of $q^D(y^{l}, y^{u}, z^{l}, z^{u})$ with respect to the
remaining variables is sought; the minimizer in the second phase may itself
need to be projected back into the dual feasible region (either
using a brute-force backtrack or a second arc search).

Both phases require the solution of sparse systems of symmetric linear
equations, and these are handled by the matrix factorization package
``SBLS`` or the conjugate-gradient package ``GLTR``.  The systems are
commonly singular, and this leads to a requirement to find the Fredholm
Alternative for the given matrix and its right-hand side.  In the
non-singular case, there is an option to update existing factorizations
using the "Schur-complement" approach given by the package ``SCU``.

Optionally, the problem may be pre-processed temporarily to eliminate dependent
constraints using the package ``FDC``. This may improve the
performance of the subsequent iteration.

reference
---------

The basic algorithm is described in

  N. I. M. Gould and D. P. Robinson,
  ``A dual gradient-projection method
  for large-scale strictly-convex quadratic problems'',
  *Computational Optimization and Applications*
  **67(1)** (2017) 1-38.
