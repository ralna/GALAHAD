.. GALAHAD C Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quadratic Programming
---------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there are eight packages designed to solve quadratic
programming problems of various kinds.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   BQP - bound-constrained quadratic programming using a preconditioned, projected-gradient method <bqp>
   BQPB - bound-constrained quadratic programming using an interior-point method <bqpb>
   CQP - convex quadratic programming using an interior-point method <cqp>
   CRO - crossover from an interior-point to basic solution for convex quadratic programming <cro>
   DQP - convex quadratic programming using a dual active-set method <dqp>
   EQP - equality-constrained quadratic programming using an iterative method <eqp>
   LSQP - linear or separable convex quadratic programming using an interior-point trust-region method <lsqp>
   QPA - non-convex quadratic programming using an active-set method <qpa>
   QPB - non-convex quadratic programming using an interior-point method <qpb>


References
^^^^^^^^^^

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
