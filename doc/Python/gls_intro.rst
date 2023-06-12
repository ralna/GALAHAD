purpose
-------

The ``gls`` package **solves sparse unsymmetric systems of linear equations**
using a variant of Gaussian elimination.
Given a sparse symmetric matrix $A = \{ a_{ij} \}_{m \times n}$, and an
$n$-vector $b$, this function solves the system $A x = b$. If instead 
$b$ is an $m$-vector, the function may solve instead $A^T x = b$.
``gls`` is based upon a modern fortran interface to the HSL Archive 
fortran 77 package ``MA28``, which itself relies on ``MA33``.
To obtain HSL Archive packages, see

- http://hsl.rl.ac.uk/archive/ .

**Currently only the options and info dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Extended functionality is available with the ``uls`` function.

See Section 4 of $GALAHAD/doc/gls.pdf for a brief description of the
method employed and other details.
