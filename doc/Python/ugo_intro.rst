purpose
-------

The ``ugo`` package aims to find the global minimizer of a univariate
twice-continuously differentiable function $f(x)$ of a single
variable over the finite interval $x^l <= x <= x^u$. Function
and derivative values are provided via a subroutine call.
Second derivatives may be used to advantage if they are available.

See Section 4 of $GALAHAD/doc/ugo.pdf for additional details.

method
------

The algorithm starts by splitting the interval $[x^l,x^u]$ into a specified
number of subintervals $[x_i,x_{i+1}]$ of equal length, and evaluating
$f$ and its derivatives at each $x_i$. A surrogate (approximating)
lower bound function is constructed on each subinterval using the
function and derivative values at each end, and an estimate of the
first- and second-derivative Lipschitz constant. This surrogate is
minimized, the true objective evaluated at the best predicted point,
and the corresponding interval split again at this point.
Any interval whose surrogate lower bound value exceeds an evaluated
actual value is discarded. The method continues until only one interval
of a maximum permitted width remains.

reference
---------

Many ingredients in the algorithm are based on the paper

  D. Lera and Ya. D. Sergeyev,
  "Acceleration of univariate global optimization algorithms working with
  Lipschitz functions and Lipschitz first derivatives"
  *SIAM J. Optimization* **23(1)**, (2013) 508–529

but adapted to use second derivatives.
