purpose
-------

Given an under-determined set of linear equations/constraints $a_i^T x =
b_i^{}$, $i = 1, \ldots, m$ involving $n \geq m$ unknowns $x$, the ``fdc``
package **determines whether the constraints are consistent, and if
so how many of the constraints are dependent**; a list of dependent
constraints, that is, those which may be removed without changing the
solution set, will be found and the remaining $a_i$ will be linearly
independent.  Full advantage is taken of any zero coefficients in the
matrix $A$ whose columns are the vectors $a_i^T$.

See Section 4 of $GALAHAD/doc/fdc.pdf for additional details.

method
------

A choice of two methods is available. In the first, the matrix
$$K = \begin{pmatrix}\alpha I & A^T \\ A & 0 \end{pmatrix}$$
is formed and factorized for some small $\alpha > 0$ using the
``SLS`` package --- the factors $K = P L D L^T P^T$ are 
used to determine whether $A$ has dependent rows. In particular, in 
exact arithmetic dependencies in $A$ will correspond to zero pivots 
in the block diagonal matrix $D$.

The second choice of method finds factors $A = P L U Q$ of the
rectangular matrix $A$ using the ``ULS`` package.  In this case,
dependencies in $A$ will be reflected in zero diagonal entries in $U$ in
exact arithmetic.

The factorization in either case may also be used to
determine whether the system is consistent.
