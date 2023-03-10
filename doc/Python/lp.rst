.. GALAHAD Python Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Linear Programming
------------------

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization.  Currently there are three packages designed to find global
optima, or at least good approximations to them.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :titlesonly:

   DLP - linear programming using a dual gradient-projection method <dlp>
   LPA - linear programming using an active-set method <lpa>
   LPB - linear programming using an interior-point method <lpb>
   WCP - find a well-centered point within a polyhedral set <wcp>


References
^^^^^^^^^^

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.
