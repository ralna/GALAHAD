purpose
-------

The ``sils`` package **solves sparse symmetric systems of linear equations**
using a multifrontal variant of Gaussian elimination.
Given a sparse symmetric matrix $A = \{ a_{ij} \}_{n \times n}$, and an
$n$-vector $b$, this function solves the system $A x = b$.
The matrix $A$ need not be definite. ``sils`` is based upon a modern fortran
interface to the HSL Archive fortran 77 package ``MA27``.
To obtain HSL Archive packages, see

- http://hsl.rl.ac.uk/archive/ .

**Currently only the options and info dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Extended functionality is available with the ``sls`` function.

See Section 4 of $GALAHAD/doc/sils.pdf for a brief description of the
method employed and other details.
