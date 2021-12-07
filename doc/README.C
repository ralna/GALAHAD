           --------------------------------
           C interfaces to GALAHAD packages
           --------------------------------

GALAHAD is a freely-available library of Fortran modules that can be 
used to solve a variety of linear and nonlinear optimization problems.

As we fully appreciate that Fortran is not the language of choice
for many practitioners, we have started to provide C interfaces 
to GALAHAD's fortran packages, using the standard Fortran ISO C bindings.
In the longer term, we hope to use these as a bridge to other 
laguages such as python and Julia.

Currently there are C interfaces to the following core packages:

  sls  - common interface to a variety of popular symmetric linear solvers
  sbls - precondition/solve symmetric block linear systems
  cqp  - solve convex quadratic programs using interior-point methods
  tru  - solve unconstrained optimization problems using trust-region methods
  arc  - solve unconstrained optimization problems using regularizatio methods
  nls  - find the smallest Euclidean norm of a vector-valued function  
  trb  - solve bound constrained optimization problems by a trust-region method

as well as the forthcoming

  ugo  - univariate global optimization
  bgo  - bound-constrained multivariate global optimization using multistart
  dgo  - bound-constrained multivariate global optimization using boundings

Documentation is available (via Doxygen) as package-specific PDF files,
html web documents and man pages. For a package named ${pack}, look at

  $GALAHAD/doc/${pack}_c.pdf
  $GALAHAD/html/C/${pack}.html
  $GALAHAD/man/man3/${pack}.3

respectively.

To use the packages, users should first build GALAHAD as usual.
The interface routine to the package named ${pack} is in

  $GALAHAD/src/${pack}/C/${pack}_iface.f90

and the associated C header file is in 

  $GALAHAD/include/${pack}.h

Examples of use (in which both C or fortran indexing are allowed) are 
provided in

  $GALAHAD/src/${pack}/C/${pack}t.c
  $GALAHAD/src/${pack}/C/${pack}tf.c

To compile, link and run these examples, issue the commands

  make -f $GALAHAD/makefiles/# ${pack}dt

(C 0-based indexing) or 

  make -f $GALAHAD/makefiles/# ${pack}dt

(fortran 1-based indexing), where # is the name of the required 
"architecture" as described in the main GALAHAD README. With luck,
this should provide a template for users' actual problems.

The list of packages supported will extend as time goes on,
but if you have a particular need, please let us know and
we will do our best to prioritise it ... some interfaces
are tricker than others!

Nick Gould  (nick.gould@stfc.ac.uk)
Jari Fowkes (jaroslav.fowkes@stfc.ac.uk)

For GALAHAD productions
7 December 2021
This version: 7 December 2021
