purpose
-------

The ``rpd`` package **reads and writes** quadratic programming
(and related) problem data to and from a QPLIB-format data file.
Variables may be continuous, binary or integer.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/rpd.pdf for additional details.

reference
---------

The QPBLIB format is defined in

  F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould,
  L. Liberti, A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis,
  S. Vigerske and A. Wiegele,
  ``QPLIB: a library of quadratic programming instances'',
  *Mathematical Programming Computation* **11** (2019) 237-–265.
