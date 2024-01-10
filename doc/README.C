           --------------------------------
           C interfaces to GALAHAD packages
           --------------------------------

Introduction
------------

GALAHAD is a freely-available library of Fortran modules that can be
used to solve a variety of linear and nonlinear optimization problems.

As we fully appreciate that Fortran may not be the language of choice
for many practitioners, we have started to provide C interfaces to
GALAHAD's fortran packages, using the standard Fortran ISO C bindings.
In the longer term, we hope to use these as a bridge to other
languages such as python and Julia.

Current interfaces
------------------

Currently there are C interfaces to the following core packages:

  uls  - common interface to a variety of popular unsymmetric linear solvers
  sls  - common interface to a variety of popular symmetric linear solvers
  psls - precondition/solve symmetric linear systems
  sbls - precondition/solve symmetric block linear systems
  fdc  - determine consistency and redundancy of linear systems
  lpa  - solve linear programs using the simplex method
  lpb  - solve linear programs using IP methods
  wcp  - find a well-centered feasible point in a polyhedron using an IP method
  blls - solve bound-constrained linear least-squares problems using projection
  bqp  - solve convex bound-constrained quadratic programs using projection
  bqpb - solve convex bound-constrained quadratic programs using IP methods
  lsqp - solve linear or seprable quadatic programs using interior-point methods
  cqp  - solve convex quadratic programs using interior-point methods
  dqp  - solve convex quadratic programs using dual projection methods
  eqp  - solve equality-constrained quadratic programs using iterative methods
  trs  - solve the trust-region subproblem using matrix factorization
  gltr - solve the trust-region subproblem using matrix-vector products
  rqs  - solve the reqularized quadratic subproblem using matrix factorization
  glrt - solve the reqularized quadratic subproblem using matrix vector prods
  dps  - solve the trust-region/regularization subproblem in diagonalizing norm
  lstr - solve the least-squares trust-region subproblem using mat-vect prods
  lsrt - solve the regularized least-squares subproblem using mat-vect prods
  l2rt - solve the regularized l_2 norm subproblem using matrix-vector prods
  qpa  - solve general quadratic programs using working-set methods
  qpb  - solve general quadratic programs using interior-point methods
  tru  - solve unconstrained optimization problems using trust-region methods
  arc  - solve unconstrained optimization problems using regularization methods
  nls  - find the smallest Euclidean norm of a vector-valued function
  trb  - solve bound constrained optimization problems by a trust-region method
  ugo  - univariate global optimization
  bgo  - bound-constrained multivariate global optimization using multistart
  dgo  - bound-constrained multivariate global optimization using boundings

Documentation
-------------

Documentation is available (via Doxygen) as package-specific PDF files,
and man pages. For a package named ${pack}, look at

  $GALAHAD/doc/${pack}_c.pdf
  $GALAHAD/man/man3/${pack}.3

respectively. HTML documentation may also be generated using the

  $GALAHAD/doc/C/gen_html_c_docs

script, and once this has been invoked, the documentation for package ${pack}
is available in

  $GALAHAD/html/C/${pack}.html

Installation
------------

To use the packages, users should first build GALAHAD as usual.
The interface routine to the package named ${pack} is in

  $GALAHAD/src/${pack}/C/${pack}_iface.f90

and the associated C header file is in

  $GALAHAD/include/${pack}.h

Examples of use (in which both C or Fortran indexing are allowed) are
provided in

  $GALAHAD/src/${pack}/C/${pack}t.c
  $GALAHAD/src/${pack}/C/${pack}tf.c

****************************************************************************
*********************** N.B. Compiler Compatibility ************************
****************************************************************************

At present, not all fortran compilers are mature enough to
suppport the full ISO-C bindings used. Currently the interfaces
compile and have been run successfully with

  gfortran (version 8.0 and above)
  ifort (2022 version)
  ifx (2024 version)
  nagfor (version 7102 and above

and we expect other compilers to catch up soon.

Linking and running
-------------------

To compile, link and run these examples, issue the commands

  make -f $GALAHAD/makefiles/# ${pack}dt

(C 0-based indexing) or

  make -f $GALAHAD/makefiles/# ${pack}dt

(Fortran 1-based indexing), where # is the name of the required
"architecture" as described in the main GALAHAD README. With luck,
this should provide a template for users' actual problems.

To link with other applications, you should use

  -lgalahad_c -lgalahad_hsl_c -lgalahad -lgalahad_hsl -lgalahad_spral \
  -lgalahad_mkl_pardiso -lgalahad_pardiso -lgalahad_wsmp \
  -lgalahad_pastix -lgalahad_mumps -lgalahad_mpi -galahad_umfpack \
  -lgalahad_metis_dummy -lgalahad_lapack -lgalahad_blas

but remember to replace any of the later libraries with vendor-specific
ones to improve performance.

Documentation
-------------

Documentation is available online from 

     https://ralna.github.io/galahad_docs/html/C

The future
----------

The list of packages supported will extend as time goes on,
but if you have a particular need, please let us know and
we will do our best to prioritise it ... some interfaces
are trickier than others!

Nick Gould  (nick.gould@stfc.ac.uk)
Jari Fowkes (jaroslav.fowkes@stfc.ac.uk)

For GALAHAD productions
7 December 2021
This version: 10 January 2024
