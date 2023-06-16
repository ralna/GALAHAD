.. GALAHAD C Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Unconstrained Optimization
--------------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there are a pair of packages designed to find a local
minimum of a function whose variables may take any values.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   ARC - unconstrained local optimization of a smooth function using a regularization method<arc>
   TRU - unconstrained local optimization of a smooth function using a trust-region method<tru>

References
^^^^^^^^^^

  .. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
