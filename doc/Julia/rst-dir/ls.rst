.. GALAHAD Julia Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Least-Squares
-------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_, `Nick Gould <nick.gould@stfc.ac.uk>`_, `Alexis Montoison <alexis.montoison@polymtl.ca>`_ and `Dominique Orban <dominique.orban@polymtl.ca>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there is a single package designed to find a local
minimizer of a sum-of-squares function whose variables may take any values,
a second pair that target linear problems with simple bounds on the variables,
another for which the feasible region is a regular simplex, and a
final one for which the constraints are linear (polyhedral).

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   BLLS - bound-constrained linear least-squares using a preconditioned, projected-gradient method <blls>
   BLLSB - bound-constrained linear least-squares using an interior-point method <bllsb>
   CLLS - linearly-constrained linear least-squares using an interior-point method <clls>
   NLS - unconstrained local nonlinear least-squares using a regularization method<nls>
   SLLS - simplex-constrained linear least-squares using a preconditioned, projected-gradient method <slls>

References
^^^^^^^^^^

  .. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
