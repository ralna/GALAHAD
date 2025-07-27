purpose
-------

Given a (possibly rectangular) matrix $A$ and symmetric matrices $H$ and $C$,
**form and factorize the block, symmetric matrix**
$$K = \begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix},$$
and susequently **solve systems**
$$\begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix} 
\begin{pmatrix}x \\ y\end{pmatrix} = 
\begin{pmatrix}a \\ b\end{pmatrix},$$
using the GALAHAD symmetric-indefinite factorization package ``SLS``
Full advantage is taken of any zero coefficients in the matrices 
$H$, $A$ and $C$.

See Section 4 of $GALAHAD/doc/ssls.pdf for additional details.

method
------

The method simply assembles $K$ from its components, and then relies
on  ``SLS`` for analysis, factorization and solves.
