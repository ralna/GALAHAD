.. GALAHAD C Interface documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GALAHAD C Interfaces
====================

**Release:** 1.0

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_ and `Nick Gould <nick.gould@stfc.ac.uk>`_

GALAHAD [1]_ is a suite of open-source routines for large-scale continuous
optimization. GALAHAD 4.1 [2]_ and above provides C functions that link
transparently to the underlying fortran.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :titlesonly:

   uco
   bco
   ls
   lp
   qp
   reg
   sys
   go
   auxiliary

* :ref:`genindex`
* :ref:`search`

References
----------

.. [1]
   Gould, N. I. M., Orban, D., & Toint, Ph. L. (2003). GALAHAD, a library of thread-safe Fortran 90 packages for large-scale nonlinear optimization. ACM Transactions on Mathematical Software (TOMS), 29(4), 353-372.

.. [2]
   Fowkes, J. M., & Gould, N. I. M. (2023). GALAHAD 4.0: an open source library of Fortran packages with C and Matlab interfaces for continuous optimization. Journal of Open Source Software 8(87), 4882.
