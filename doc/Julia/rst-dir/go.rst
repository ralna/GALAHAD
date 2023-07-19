.. GALAHAD Julia Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Global Optimization
-------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_, `Nick Gould <nick.gould@stfc.ac.uk>`_, `Alexis Montoison <alexis.montoison@polymtl.ca>`_ and `Dominique Orban <dominique.orban@polymtl.ca>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there are three packages designed to find global
optima, or at least good approximations to them.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   UGO - univariate global optimization of a smooth function over an interval <ugo>
   BGO - multivariate global optimization of a smooth function within a box using stochastic methods<bgo>
   DGO - multivariate global optimization of a smooth function within a box using deterministic methods<dgo>

References
^^^^^^^^^^

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
