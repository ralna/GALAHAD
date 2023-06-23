---
title: '`GALAHAD 4.0`: an open source library of Fortran packages
with C and Matlab interfaces for continuous optimization'
tags:
  - C
  - Fortran
  - Matlab
  - optimization
authors:
  - name: Jaroslav M. Fowkes
    affiliation: 1
  - name: Nicholas I. M. Gould
    affiliation: 1
affiliations:
  - name: Science and Technology Facilities Council, Rutherford Appleton Laboratory, Harwell Campus, Didcot, Oxfordshire, OX11 0QX, UK
    index: 1
date: June 2023
bibliography: paper.bib
---
# Summary

The ability to solve continuous optimization problems is one of the
cornerstones of computational mathematics. Such problems occur
throughout science, engineering, planning and economics, since nature
(and mankind) loves to optimize (minimize or maximize). Most real-life
models of physical phenomena are nonlinear, and when discretised for
computer solution they usually involve a large number of minimization
variables (parameters) and/or constraints. Thus it is valuable to be
able to rely on software specifically designed for optimization, and
particularly that is designed to solve large, nonlinear problems.

Continuous optimization problems occur in a variety of formats. Problems
may or may not involve constraints, least-squares fitting being a common
but vital example of the latter. If there are constraints, they may
simply be bounds on the values of the variables, or there may be linear
or nonlinear relationships (both equations or inequalities) between sets
of variables. In an ideal world, a global optimizer is sought, but often
that is beyond current (and likely future) expectations particularly if
there are a large number of variables involved; fortunately a local
minimizer often suffices. There is also a natural hierarchy of problems,
and the ability to solve one is useful if it occurs as a subproblem in a
harder one---solving linear systems (sometimes approximately) is vital
in linear or quadratic programming, quadratic programs are used within
nonlinear programming methods, and local optimization is often a vital
component of global optimization.

Thus ideally a comprehensive optimization library should address the
different needs of its users by providing software tuned to a variety of
commonly-occurring subclasses of problems. This is the aim of
[`GALAHAD`](https://github.com/ralna/GALAHAD). `GALAHAD` provides
packages for basic subproblem solvers (such as for linear systems,
trust-region and regularization of quadratic and linear least-squares
functions), linear and quadratic programming, unconstrained and
bound-constrained optimization, nonlinear least-squares fitting, general
nonlinear programming and both approximate univariate and multivariate
global optimization, together with an array of attendant utilities packages
(such as for polynomial fitting, hashing, presolves, and matrix
approximation).  It is also recognised that there are excellent external
sources of relevant software, particular for solving linear systems,
and `GALAHAD` provides uniform bridges to these if they are available.

# Statement of need

The first release of the Fortran 90 `GALAHAD` library [@GoulOrbaToin03]
aimed to  expand the functionality of the earlier Fortran 77
`LANCELOT` package [@ConnGoulToin92] for nonlinear optimization.
Subsequent releases focused on increasing the scope of solvers provided,
but aside from limited interfaces to Matlab and to the
[`CUTEst`](https://github.com/ralna/CUTEst) modeling library
[@GoulOrbaToin15], little effort was made to
bridge the gap between Fortran and other, often more recent and popular,
programming languages. `GALAHAD 4.0` addresses this deficiency.

Although `GALAHAD 4.0` contains an increased variety of new solvers, the
principal motivation for the new release is to raise the profile of the
library by increasing its potential userbase. While modern Fortran is an
extremely flexible programming language, it is perceived as old fashioned
in many circles. Rival open-source solvers such as
[`IPOPT`](https://github.com/coin-or/Ipopt) [@WachBieg06] and
commercial ones such as
[`KNITRO`](https://www.artelys.com/docs/knitro/) [@ByrdNoceWalt06]
are written predominantly in C/C++, and this is attractive as there are
often straightforward bridges from C to other popular languages such as
Python and Julia. Thus, we have now provided interfaces between Fortran and C
for a significant subset of the Fortran packages. This has been
made possible using the standardised ISO-C bindings introduced in
Fortran 2003, and enhanced in more modern revisions. Essentially an
interface program binds Fortran types and functions to C equivalents, and
a second C header file provides the C access.

A current list of major packages with C interfaces and their functionality
is as follows:

| package | purpose |
| :-------| ------- |
| uls | unsymmetric linear systems (external bridge) |
| sls | symmetric linear systems (external bridge) |
| sbls | symmetric block linear systems |
| psls | preconditioners for symmetric linear systems |
| fdc | determine consistency and redundancy of linear systems |
| lpa | linear programming using an active-set method (external bridge) |
| lpb | linear programming using an interior-point method |
| wcp | linear feasibility using an interior-point method |
| blls | bound-constrained linear least-squares problems using a gradient-projection method |
| presolve| simplify quadratic programs prior to solution |
| bqp | bound-constrained convex quadratic programming using a gradient-projection method |
| bqpb | bound-constrained convex quadratic programming using an interior-point method |
| lsqp | linear and separable quadratic programming using an interior-point method |
| cqp | convex quadratic programming using an interior-point method |
| dqp | convex quadratic programming using a dual active-set method |
| eqp | equality-constrained quadratic programming using an iterative method |
| trs | the trust-region subproblem using matrix factorization |
| gltr | the trust-region subproblem using matrix-vector products |
| rqs | the regularized quadratic subproblem using matrix factorization |
| glrt | the regularized quadratic subproblem using matrix-vector products |
| dps | the trust-region and regularized quadratic subproblems in a diagonalising norm |
| llsr | the regularized linear least-squares subproblem matrix factorization |
| llst | the linear least-squares trust-region subproblem matrix factorization |
| lstr | the linear least-squares trust-region subproblem using matrix-vector products |
| lsrt | the regularized linear least-squares subproblem using matrix-vector products |
| l2rt | the regularized linear $l_2$ norm subproblem using matrix-vector products |
| qpa | general quadratic programming using an active-set method |
| qpb | general quadratic programming using an interior-point method |
| blls | bound-constrained linear-least-squares using a gradient-projection method |
| tru | unconstrained optimization using a trust-region method |
| arc | unconstrained optimization using a regularization method |
| nls | least-squares optimization using a regularization method |
| trb | bound-constrained optimization using a gradient-projection trust-region method |
| ugo | univariate global optimization |
| bgo | multivariate global optimization in a box using a multi-start trust-region method |
| dgo | multivariate global optimization in a box using a deterministic partition-and-bound method |

Interfaces to other `GALAHAD` packages, such as `LANCELOT` and `FILTRANE`
will be provided as time and demand permit. Future extensions to provide
follow-on interfaces to Python and Julia using the C functionality are underway.

`GALAHAD` is easy to install using its own make -based system. Fortran
documentation is provided in PDF via LaTeX, while HTML, PDF and man
documents for the C packages are available using Doxygen, Doxyrest and Sphinx.

# Acknowledgements

The authors are grateful for support provided by the Engineering and Physical
Sciences Research Council (UK) grants EP/F005369/1, EP/I013067/1 and
EP/M025179/1.

# References
