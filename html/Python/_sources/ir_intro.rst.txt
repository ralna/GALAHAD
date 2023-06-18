purpose
-------

Given a sparse symmetric $n \times n$ matrix $A = a_{ij}$ and the 
factorization of $A$ found by the GALAHAD package SLS, the ``ir`` package 
**solves the system of linear equations $A x = b$ using
iterative refinement.**

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/ir.pdf for additional details.

method
------

Iterative refinement proceeds as follows. First obtain the floating-point
solution to $A x = b$ using the factors of $A$. Then iterate
until either the desired residual accuracy (or the iteration limit is
reached) as follows: evaluate the residual $r = b - A x$,
find the floating-point solution $\delta x$ to $A \delta x = r$,
and replace $x$ by $x + \delta x$.
