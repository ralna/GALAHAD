.. GALAHAD Python Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Auxiliary Procedures
--------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  This is supported by a number of auxiliary procedures that
are used to perform commonly-occurring numerical tasks.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   BSC - build and use the Schur complement from constituent matrices <bsc>
   CONVERT - convert a sparse matrix from one format to another <convert> 
   FIT - fit function and derivative values to data <fit>
   HASH - set up and use a chained scatter table <hash>
   IR - given matrix factors, perform iterative refinement to solve systems <ir>
   LHS - compute an array of Latin Hypercube samples <lhs> 
   LMS - maintain limited-memory Hessian approximations <lms> 
   ROOTS - find real roots of real polynomials <roots> 
   RPD - convert LP/QP data to and from QPLIB format <rpd> 
   SCU - build and extend factors for an evolving block sparse matrix <scu> 
   SEC - maintain dense BFGS and SR1 secant approximations to a Hessian <sec> 
   SHA - find a sparse Hessian matrix approximation using componentwise secant approximation <sha> 
   PRESOLVE - transform LP/QP data so that the resulting problem is easier to solve <presolve> 

References
^^^^^^^^^^

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
