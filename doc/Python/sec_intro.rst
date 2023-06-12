purpose
-------

The ``sec`` package 
**builds and updates dense BFGS and SR1 secant approximations to a Hessian**
so that the approximation $B$ satisfies the secant condition $B s = y$
for given vectors $s$ and $y$.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/sec.pdf for a brief description of the
method employed and other details.
