purpose
-------

The ``scu`` package computes the **solution to an extended system of $n + m$
sparse real linear equations in $n + m$ unknowns,**
$$\begin{pmatrix}A & B \\ C  & D\end{pmatrix}
\begin{pmatrix}x_1 \\ x_2\end{pmatrix} = 
\begin{pmatrix}b_1 \\ b_2\end{pmatrix}$$
in the case where the $n$ by $n$ matrix $A$ is nonsingular
and solutions to the systems
$$A x  =  b \;\;\mbox{and}\;\; A^T y  =  c$$
may be obtained from an external source, such as an existing
factorization.  The subroutine uses reverse communication to obtain
the solution to such smaller systems.  The method makes use of
the Schur complement matrix
$$S = D - C A^{-1} B.$$
The Schur complement is stored and factorized as a dense matrix
and the subroutine is thus appropriate only if there is
sufficient storage for this matrix. Special advantage is taken
of symmetry and definiteness in the coefficient matrices.
Provision is made for introducing additional rows and columns
to, and removing existing rows and columns from, the extended.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/scu.pdf for additional details.

method
------

The function ``scu_factorize`` forms the Schur complement 
$S$ of $A$ in the extended matrix by repeated 
reverse communication to obtain the columns of  $A^{-1} B$. 
The Schur complement or its negative is then factorized 
into its *QR* or, if possible, Cholesky factors. 

The function ``scu_solve`` solves the extended system using 
the following well-known scheme: 

* **(i)** Compute the solution to $A u = b_1$; 

* **(ii)** Compute $x_2$ from $S x_2 = b_2 - C u$; 

* **(iii)** Compute the solution to $A v = B x_2$; and 

* **(iv)** Compute $x_1 = u - v$. 

The functions ``scu__append`` and ``scu_delete`` compute the
factorization of the Schur complement after a row and column have been
appended to, and removed from, the extended matrix, respectively.
The existing factorization is updated to obtain the new one; this is
normally more efficient than forming the factorization from scratch.
