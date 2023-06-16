.. GALAHAD C Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Regularization subproblems
--------------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there ten packages designed to find a global
minimizer of a variety of regularized quadratic functions

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   TRS - global minization of a quadratic function within an ellipsoid using matrix factorization <trs>
   RQS - global minization of a regularized quadratic function using matrix factorization <rqs>
   DPS - global minization of a regularized quadratic function in a diagonalising norm using matrix factorization <dps>
   GLTR - global minization of a quadratic function within an ellipsoid using an iterative method <gltr>
   GLRT - global minization of a regularized quadratic function <glrt>
   LLSR - global minization of a regularized linear least-squares objective using matrix factorizations <llsr>
   LLST - global minization of a linear least-squares objective within an ellipsoid using matrix factorizations <llst>
   LSTR - global minization of a linear least-squares objective within a sphere using an iterative method <lstr>
   LSRT - global minization of a regularized linear least-squares objective using an iterative method <lsrt>
   L2RT - global minization of a regularized linear least-Euclidean-norm objective using an iterative method <l2rt>


References
^^^^^^^^^^

  .. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
