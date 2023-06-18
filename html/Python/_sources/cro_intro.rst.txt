purpose
-------

The ``cro`` package provides a **crossover** from a
**primal-dual interior-point**
solution to given **convex quadratic program** to a **basic one** in which 
the matrix of defining active constraints/variables is of **full rank**. 
This applies to the problem of minimizing the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric postive-semi-definite and $m$ by $n$ matrices,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method is most suitable for problems involving a large number of 
unknowns $x$.

See Section 4 of $GALAHAD/doc/cro.pdf for additional details.

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
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 
\;\;\mbox{and}\;\; z_u \leq 0,\;\;\mbox{(1)}$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

Denote the active constraints by $A_A x = c_A$ and the
active bounds by $I_A x = x_A$. Then any optimal solution satisfies
the linear system
$$\left(\begin{array}{ccc}
H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 
\end{array}\right) \left(\begin{array}{c} 
x \\ y_A \\ z_A
\end{array}\right) = \left(\begin{array}{c} 
- g \\ c_A \\ x_A 
\end{array}\right), $$
where $y_A$ and $z_A$ are the corresponding active Lagrange multipliers
and dual variables respectively. Consequently the difference between
any two solutions $(\Delta x, \Delta y, \Delta z)$ must satisfy
$$\left(\begin{array}{ccc}
H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 
\end{array}\right) \left(\begin{array}{c}
\Delta x \\ \Delta y_A \\ \Delta z_A
\end{array}\right) = 0.\;\;\mbox{(2)}$$
Thus there can only be multiple solution if the coefficient matrix $K$
of (2) is singular. The algorithm used in \fullpackagename\
exploits this. The matrix $K$ is checked for singularity
using the package ``ULS``. If $K$ is
non singular, the solution is unique and the solution input by the user
provides a linearly independent active set. Otherwise $K$ is singular,
and partitions $A_A^T = ( A_{AB}^T \;\; A_{AN}^T)$ and
$I_A^T = ( I_{AB}^T \;\; I_{AN}^T)$ are found so that
$$\left(\begin{array}{ccc}
H & - A_{AB}^T & - I^T_{AB} \\ A_{AB} & 0 & 0 \\  I_{AB} & 0 & 0 
\end{array}\right)$$
is non-singular and the *non-basic* constraints $A_{AN}^T$
and $I_{AN}^T$ are linearly dependent on the *basic* ones
$( A_{AB}^T \;\; I_{AB}^T)$. In this case (2) is equivalent to
$$\left(\begin{array}{ccc}
H & - A_{AB}^T & - I^T_{AB} \\ A_{AB} & 0 & 0 \\ I_{AB} & 0 & 0 
\end{array}\right) \left(\begin{array}{c}
\Delta x \\ \Delta y_{AB} \\ \Delta z_{AB} 
\end{array}\right) = \left(\begin{array}{c}
A_{AN}^T \\ 0 \\ 0 
\end{array}\right) \Delta y_{AN} + \left(\begin{array}{c}
I^T_{AN} \\ 0 \\ 0 
\end{array}\right) \Delta z_{AN} .\;\;\mbox{(3)}$$
Thus, starting from the user's $(x, y, z)$
and with a factorization of the coefficient matrix of (3)
found by the package ``SLS``,
the alternative solution $(x + \alpha x, y + \alpha y, z + \alpha z)$,
featuring $(\Delta x, \Delta y_{AB}, \Delta z_{AB})$ from (3)
in which successively one of the components of $\Delta y_{AN}$
and $\Delta z_{AN}$ in turn is non zero, is taken.
The scalar $\alpha$ at each stage
is chosen to be the largest possible that guarantees (1);
this may happen when a non-basic multiplier/dual variable reaches zero,
in which case the corresponding constraint is disregarded, or when this
happens for a basic multiplier/dual variable, in which case this constraint is
exchanged with the non-basic one under consideration and disregarded.
The latter corresponds to changing the basic-non-basic partition
in (3), and subsequent solutions may be found by updating
the factorization of the coefficient matrix in (3)
following the basic-non-basic swap using the package ``SCU``.
