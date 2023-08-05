purpose
-------

Given a **block, symmetric matrix**
$$K_{H} = \begin{pmatrix}H & A^T \\ A  & - C\end{pmatrix},$$
the ``sbls`` package constructs a variety of **preconditioners** of the form
$$K_{G} = \begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix}.$$
Here, the leading-block matrix $G$ is a suitably-chosen
approximation to $H$; it may either be prescribed **explicitly**, in
which case a symmetric indefinite factorization of $K_{G}$
will be formed using the GALAHAD package ``SLS``,
or **implicitly**, by requiring certain sub-blocks of $G$
be zero. In the latter case, a factorization of $K_{G}$ will be
obtained implicitly (and more efficiently) without recourse to ``sls``.
In particular, for implicit preconditioners, a reordering
$$K_{G} = P \begin{pmatrix}
G_{11}^{} & G_{21}^T & A_1^T \\ 
G_{21}^{} & G_{22}^{} & A_2^T \\ A_{1}^{} & A_{2}^{} & - C
\end{pmatrix} P^T$$
involving a suitable permutation $P$ will be found, for some
invertible sub-block ("basis") $A_1$ of the columns of $A$;
the selection and factorization of $A_1$ uses the package ``uls``.
Once the preconditioner has been constructed,
solutions to the preconditioning system
$$\begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix} 
\begin{pmatrix}x \\ y\end{pmatrix} = 
\begin{pmatrix}a \\ b\end{pmatrix}$$
may be obtained by the package. Full advantage is taken of any zero 
coefficients in the matrices $H$, $A$ and $C$.

See Section 4 of $GALAHAD/doc/sbls.pdf for additional details.

method
------

The method used depends on whether an explicit or implicit
factorization is required. In the explicit case, the
package is really little more than a wrapper for the
symmetric, indefinite linear solver ``SLS`` in
which the system matrix $K_G$ is assembled from its constituents
$A$, $C$ and whichever $G$ is requested by the user.
Implicit-factorization preconditioners are more involved,
and there is a large variety of different possibilities. The
essential ideas are described in detail in

  H. S. Dollar, N. I. M. Gould and A. J. Wathen.
  ``On implicit-factorization constraint preconditioners''.
  In  Large Scale Nonlinear Optimization (G. Di Pillo and M. Roma, eds.)
  Springer Series on Nonconvex Optimization and Its Applications, Vol. 83,
  Springer Verlag (2006) 61--82

and

  H. S. Dollar, N. I. M. Gould, W. H. A. Schilders and A. J. Wathen
  ``On iterative methods and implicit-factorization preconditioners for
  regularized saddle-point systems''.
  *SIAM Journal on Matrix Analysis and Applications* **28(1)** (2006) 170--189.

The range-space factorization is based upon the decomposition
$$K_G = \begin{pmatrix}G & 0 \\ A & I\end{pmatrix} 
\begin{pmatrix}G^{-1} & 0 \\ 0 & - S\end{pmatrix} 
\begin{pmatrix}G & A^T \\ 0 & I\end{pmatrix},$$
where the ``Schur complement'' $S = C + A G^{-1} A^T$.
Such a method requires that $S$ is easily invertible. This is often the
case when $G$ is a diagonal matrix, in which case $S$ is frequently
sparse, or when $m \ll n$ in which case $S$
is small and a dense Cholesky factorization may be used.

When $C = 0$, the null-space factorization is based upon the decomposition
$$K_{G} = P \begin{pmatrix}
G_{11}^{} & 0 & I \\ G_{21}^{} & I & A_{2}^{T} A_{1}^{-T} \\ A_{1}^{} & 0 & 0
\end{pmatrix} \begin{pmatrix}
0 & 0 & I \\ \;\;\; 0 \;\; & \;\; R \;\; & 0 \\ I & 0 & - G_{11}^{}
\end{pmatrix} \begin{pmatrix}
G_{11}^{} & G_{21}^T & A_{1}^T \\ 0 & I & 0 \\ I & A_{1}^{-1} A_{2}^{} & 0
\end{pmatrix} P^T,$$
where the ``reduced Hessian''
$$R = ( - A_{2}^{T} A_1^{-T} \;\; I ) 
\begin{pmatrix}G_{11}^{} & G_{21}^{T} \\ G_{21}^{} & G_{22}^{}\end{pmatrix} 
\begin{pmatrix}- A_1^{-1} A_2^{} \\ I\end{pmatrix}$$
and $P$ is a suitably-chosen permutation for which $A_1$ is invertible.
The method is most useful when $m \approx n$ as then the dimension
of $R$ is small and a dense Cholesky factorization may be used.
