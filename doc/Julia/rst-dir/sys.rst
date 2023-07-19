.. GALAHAD Julia Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Linear Systems
--------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_, `Nick Gould <nick.gould@stfc.ac.uk>`_, `Alexis Montoison <alexis.montoison@polymtl.ca>`_ and `Dominique Orban <dominique.orban@polymtl.ca>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there are seven packages designed to provide
preconditioners and solve linear systems of equations.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   SLS - solve symmetric systems of linear equations (übersolver) <sls>
   ULS - solve unsymmetric systems of linear equations (übersolver) <uls>
   SBLS - precondition and solve block symmetric systems of linear equations <sbls>
   PSLS - precondition symmetric, positive-definite systems of linear equations (übersolver) <psls>
   SILS - solve symmetric systems of linear equations <sils>
   GLS - solve unsymmetric systems of linear equations <gls>
   FDC - find an equivalent linearly independent subset of a system of linear equations <fdc>


References
^^^^^^^^^^

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
