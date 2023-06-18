purpose
-------

The ``roots`` package uses classical formulae together with Newton’s method 
to **find all the real roots of a real polynomial.**

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/roots.pdf for additional details.

method
------

Littlewood and Ferrari's algorithms are used to find estimates of the
real roots of cubic and quartic polynomials, respectively; a stabilized version 
of the well-known formula is used in the quadratic case. Newton's
method and/or methods based on the companion matrix are used to further 
refine the computed roots if necessary. Madsen and Reid's (1975) 
method is used for polynomials whose degree exceeds four.

reference
---------

The basic method is that given by

  K. Madsen and J. K. Reid, 
  ``FORTRAN Subroutines for Finding Polynomial Zeros''.
  Technical Report A.E.R.E. R.7986, Computer Science and System Division, 
  A.E.R.E. Harwell, Oxfordshire (1975)
