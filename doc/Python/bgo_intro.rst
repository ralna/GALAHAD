purpose
-------

The ``bgo`` package uses a multi-start trust-region method to find an
approximation to the global minimizer of a differentiable objective
function $f(x)$ of n variables $x$, subject to simple
bounds $x^l <= x <= x^u$ on the variables. Here, any of the
components of the vectors of bounds $x^l$ and $x^u$
may be infinite. The method offers the choice of direct and
iterative solution of the key trust-region subproblems, and
is suitable for large problems. First derivatives are required,
and if second derivatives can be calculated, they will be exploited -
if the product of second derivatives with a vector may be found but
not the derivatives themselves, that may also be exploited.

The package offers both random multi-start and local-minimize-and-probe
methods to try to locate the global minimizer. There are no theoretical
guarantees unless the sampling is huge, and realistically the success
of the methods decreases as the dimension and nonconvexity increase.

See Section 4 of $GALAHAD/doc/bgo.pdf for additional details.

method
------

A choice of two methods is available. 
In the first, local-minimization-and-probe, approach, local minimization
and univariate global minimization are intermixed. Given a current 
champion $x^S_k$, a local minimizer $x_k$ of $f(x)$ within the
feasible box $x^l \leq x \leq x^u$ is found using ``TRB``.
Thereafter $m$ random directions $p$ are generated, and univariate
local minimizer of $f(x_k + \alpha p)$ as a function of the scalar
$\alpha$ along each $p$ within the interval $[\alpha^L,\alpha^u]$,
where $\alpha^L$ and $\alpha^u$ are the smallest and largest $\alpha$ 
for which $x^l \leq x_k + \alpha p \leq x^u$, 
is performed using ``UGO``. The point $x_k + \alpha p$
that gives the smallest value of $f$ is then selected as the new champion
$x^S_{k+1}$.

The random directions $p$ are chosen in one of three ways. The simplest is
to select the components as
$$p_i = \mbox{pseudo random $\in$} \left\{ 
\begin{array}{rl} 
\mbox{[-1,1]} & \mbox{if} \;\; x^l_i < x_{k,i} < x^u_i \\
\mbox{[0,1]} & \mbox{if} \;\; x_{k,i}  = x^l_i \\
\mbox{[-1,0]} & \mbox{if} \;\;  x_{k,i}  = x^u_i
\end{array}
\right.$$
for each $1 \leq i \leq n$. An alternative is to 
pick $p$ by partitioning each dimension of the feasible "hypercube" box 
into $m$ equal segments, and then selecting sub-boxes randomly within 
this hypercube using Latin hypercube sampling via ``LHS``.
Each components of $p$ is then selected in its sub-box, either uniformly
or pseudo randomly.

The other, random-multi-start, method provided selects $m$ starting points 
at random, either componentwise pseudo randomly in the feasible box, 
or by partitioning each component into $m$ equal segments, assigning each to
a sub-box using Latin hypercube sampling, and finally choosing the
values either uniformly or pseudo randomly. Local minimizers within the
feasible box are then computed by ``TRB``, and
the best is assigned as the current champion. This process is then
repeated until evaluation limits are achieved.

If $n=1$, ``UGO`` is called directly.

We reiterate that there are no theoretical guarantees unless the sampling 
is huge, and realistically the success of the methods decreases as the 
dimension and nonconvexity increase. Thus the methods used should best
be viewed as heuristics.

references
----------

The generic bound-constrained trust-region method is described in detail in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint,
  Trust-region methods.
  SIAM/MPS Series on Optimization (2000),

the univariate global minimization method employed is an extension of that
due to

  D. Lera and Ya. D. Sergeyev,
  ``Acceleration of univariate global optimization algorithms working with
  Lipschitz functions and Lipschitz first derivatives''
  *SIAM J. Optimization* **23(1)** (2013) 508–529,

while the Latin-hypercube sampling method employed is that of

  B. Beachkofski and R. Grandhi,
  ``Improved Distributed Hypercube Sampling'',
  43rd AIAA structures, structural dynamics, and materials conference,
  (2002) 2002-1274.
