.. _doxid-index_rpd:

GALAHAD RPD package
===================

.. module:: galahad.rpd

.. include:: ../../Python/rpd_intro.rst

.. include:: rpd_storage.rst

.. toctree::
	:hidden:

	rpd_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the rpd package must be 
called in the following order:

* :ref:`rpd_initialize <doxid-galahad__rpd_8h_1a6805ebb5cc097db7df39723c64cef793>` - provide default control parameters and set up initial data structures

* :ref:`rpd_get_stats <doxid-galahad__rpd_8h_1ad0148374adcd7bf5f34f378ba0995a21>` - read a given QPLIB file into internal data structures, and report vital statistics

* (optionally, and in any order, where relevant)
  
  * :ref:`rpd_get_g <doxid-galahad__rpd_8h_1aa5be687c00e4a7980c5ea7c258717d3a>` - get the objective gradient term $g$
  
  * :ref:`rpd_get_f <doxid-galahad__rpd_8h_1a38dc68ed79b192e3fcd961b8589d202c>` - get the objective constant term $f$
  
  * :ref:`rpd_get_xlu <doxid-galahad__rpd_8h_1a6a5cbf68b561cc6db0ba08304d28787c>` - get the variable bounds $x_l$ and $x_u$
  
  * :ref:`rpd_get_xlu <doxid-galahad__rpd_8h_1a6a5cbf68b561cc6db0ba08304d28787c>` - get the constraint bounds $c_l$ and $c_u$
  
  * :ref:`rpd_get_h <doxid-galahad__rpd_8h_1a02021324df6f485160d327f2f5fca0d3>` - get the objective Hessian term $H$
  
  * :ref:`rpd_get_a <doxid-galahad__rpd_8h_1a8b0c3c507b12512b09ee4ec92596148e>` - get the constrain Jacobian term $A$
  
  * :ref:`rpd_get_h_c <doxid-galahad__rpd_8h_1a55ae091188ad0d88920565549bd47451>` - get the constraint Hessian terms $H_c$
  
  * :ref:`rpd_get_x_type <doxid-galahad__rpd_8h_1af784ecc65c925575788a494bd8118f4d>` - determine the type of each variable $x$
  
  * :ref:`rpd_get_x <doxid-galahad__rpd_8h_1afbc831595295e9153e4740d852a35c27>` - get initial value of the variable $x$
  
  * :ref:`rpd_get_y <doxid-galahad__rpd_8h_1ac9fd1a08acf460b7962ad5393d69fff5>` - get initial value of Lagrange multipliers $y$
  
  * :ref:`rpd_get_z <doxid-galahad__rpd_8h_1ab1579a81766096bd1764f0fb0cc10db3>` - get initial value of the dual variables $z$

* :ref:`rpd_terminate <doxid-galahad__rpd_8h_1af49fc46839c605dd71d2666189d0d8a9>` - deallocate data structures

See the :ref:`examples <doxid-index_rpd_examples>` section for 
illustrations of use.

.. include:: irt.rst

.. include:: rpd_functions.rst

available structures
--------------------

.. include :: struct_rpd_control_type.rst

.. include :: struct_rpd_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_rpd_examples:

example calls
-------------

This is an example of how to use the package to read and write a QP;
the code is available in $GALAHAD/src/rpd/Julia/test_rpd.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/rpd/Julia/test_rpd.jl
   :code: julia
