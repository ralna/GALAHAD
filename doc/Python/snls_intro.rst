purpose
-------

The ``snls`` package uses a **regularization method** to solve
a given **simplex-constrained nonlinear least-squares problem**.
The aim is to minimize the least-squares objective function
$$f(x) :=
\frac{1}{2} \sum_{i=1}^{m_r} w_i r_i^2(x) \equiv \frac{1}{2} \|r(x)\|^2_W$$
where the variables $x$ are required to lie within the
**regular simplex**
$$e^T x = 1 \;\;\mbox{and}\;\; x \geq 0,$$
or an intersection of **multiple non-overlapping regular simplices**
$$e_{\cal C_i}^T x_{\cal C_i}^{} = 1 \;\;\mbox{and}\;\; x_{\cal C_i}^{}
\geq 0 \;\;\mbox{for}\;\; i = 1,\ldots,m,$$
where the $o$ by $n$ real **design matrix** $A_o$, the vector $b$
of **observations**
and the non-negative **weights** $w$ are given, $e$ is the vector of ones, 
the vector $v_{\cal C}$ is made up of those entries of $v$ indexed by
the set $\cal C$, 
the index sets of **cohorts** $\cal C_i \subseteq \{1,\ldots,n\}$
for which $\cal C_i \cap \cal C_j = \emptyset$ for $1 \leq i, \neq j \leq m$,
and where the weighted-Euclidean norms
is given by $\|v\|_W^2 = v^T W v$ with $W = \mbox{diag}(w)$.
The method offers the choice of projected-gradient and interior-point
solution of the key regularization subproblems, and is most suitable 
for problems involving a large number of unknowns $x$.
First derivatives of the residual function $r(x)$ are required, and if
second derivatives of the $r_i(x)$ can be calculated, they may be exploited.

See Section 4 of $GALAHAD/doc/snls.pdf for additional details.

terminology
-----------

The algorithm used by the package is iterative. From the current best estimate
of the minimizer $x_k$, a trial improved point $x_k + s_k$ is sought.
The correction $s_k$ is chosen to improve a model $m_k(s)$ of
the objective function $f(x_k+s)$ built around
$x_k$. The model is the sum of two basic components,
a suitable approximation $t_k(s)$ of $f(x_k+s)$,
and a regularization term $\frac{1}{2} \sigma_k \|s\|_2^2$
involving a weight $\sigma_k$. The weight is adjusted as 
the algorithm progresses to ensure convergence.

The model $t_k(s)$ is a truncated Taylor-series approximation, and this
relies on being able to compute or estimate derivatives of $c(x)$.
Various models are provided, and each has different derivative requirements.
We denote the $m$ by $n$ residual **Jacobian**
$J(x) \equiv \nabla_x c(x)$ as the matrix  whose $i,j$-th component
$$J(x)_{i,j} := \partial r_i(x) / \partial x_j \;\;
\mbox{for $i=1,\ldots,m$ and $j=1,\ldots,n$.}$$
For a given $m$-vector $y$, the **weighted-residual Hessian** is the sum
$$H(x,y) := \sum_{\ell=1}^m y_{\ell} H_{\ell}(x), \;\; \mbox{where}\;\; H_{\ell}(x)_{i,j} := \partial^2 r_{\ell}(x) / \partial x_i \partial x_j \;\; \mbox{for $i,j=1,\ldots,n$}$$
is the Hessian of $r_\ell(x)$.
The models $t_k(s)$ provided are,

1. the **Gauss-Newton** approximation
   $\frac{1}{2} \| r(x_k) + J(x_k) s\|^2_W$,

2. the **Newton (second-order Taylor)** approximation

   $f(x_k) + g(x_k)^T s + \frac{1}{2} s^T [ J^T(x_k) W J(x_k) + H(x_k,W r(x_k))] s$, and


method
------

An adaptive regularization method is used.
In this, an improvement to a current
estimate of the required minimizer, $x_k$ is sought by computing a
step $s_k$. The step is chosen to approximately minimize a model $t_k(s)$
of $f_{\rho,r}(x_k+s)$
that includes a weighted regularization term
$\frac{\sigma_k}{p} \|s\|_{S_k}^p$
for some specified positive weight $\sigma_k$. The quality of the
resulting step $s_k$ is assessed by computing the "ratio"
$(f(x_k) - f(x_k + s_k))/(t_k(0) - t_k(s_k))$.
The step is deemed to have succeeded if the ratio exceeds a given $\eta_s > 0$,
and in this case $x_{k+1} = x_k + s_k$. Otherwise
$x_{k+1} = x_k$, and the weight is increased by powers of a given
increase factor up to a given limit. If the ratio is larger than
$\eta_v \geq \eta_d$, the weight will be decreased by powers of a given
decrease factor again up to a given limit. The method will terminate
as soon as $f(x_k)$ or
$\|\nabla_x f(x_k)\|$ is smaller than a specified value.

The step $s_k$ may be computed either by employing a projected-gradient
method to minimize the model within the simplex constraint set 
$\calC(x_k+s)$ using the \galahad\ module `slls`, or by
applying the interior-point method available in the module `sllsb`
to the same subproblem. Experience has shown that it can be beneficial
to use the latter method during early iterations, but to switch to the 
former as the iterates approach convergence.


references
----------

The generic adaptive cubic regularization method is described in detail in

  C. Cartis,  N. I. M. Gould and Ph. L. Toint,
  ''Evaluation complexity of algorithms for nonconvex optimization''
  SIAM-MOS Series on Optimization (2022),

and uses ''tricks'' as suggested in

  N. I. M. Gould, M. Porcelli and Ph. L. Toint,
  ''Updating the regularization parameter in the adaptive cubic regularization
  algorithm''.
  *Computational Optimization and Applications*
  **53(1)** (2012) 1--22.

The specific methods employed here are discussed in

  N. I. M. Gould et. al.,
  ''Nonlinear least-squares over unit simplices''.
  Rutherford Appleton Laboratory, Oxfordshire, England (2026) in preparation.
