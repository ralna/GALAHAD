.. _doxid-index_dqp:

GALAHAD DQP package
===================

.. include:: ../../Python/dqp_intro.rst

.. include:: ah_storage.rst

.. toctree::
	:hidden:

	dqp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the dqp package must be called in the following order:

* :ref:`dqp_initialize <doxid-galahad__dqp_8h_1a19aea950ca15a63e11702af3b4e777a2>` - provide default control parameters and set up initial data structures

* :ref:`dqp_read_specfile <doxid-galahad__dqp_8h_1a1db755c043c56f0afdc8e61c8ebfc517>` (optional) - override control values by reading replacement values from a file

* :ref:`dqp_import <doxid-galahad__dqp_8h_1a126153a2c845e1840b01cbd28a5b187d>` - set up problem data structures and fixed values

* :ref:`dqp_reset_control <doxid-galahad__dqp_8h_1abff958ca01c88bb049bd06a238dcbefe>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of

  * :ref:`dqp_solve_qp <doxid-galahad__dqp_8h_1a2b72dcb3fe12c15b79be741304583bd4>` - solve the quadratic program

  * :ref:`dqp_solve_sldqp <doxid-galahad__dqp_8h_1a5175cda6f83c34b45115527af26f9da2>` - solve the shifted least-distance problem

* :ref:`dqp_information <doxid-galahad__dqp_8h_1ae0ac5195553f6dcccc43f53f0e08b0a4>` (optional) - recover information about the solution and solution process

* :ref:`dqp_terminate <doxid-galahad__dqp_8h_1ac2f0f150bf38c9cc0ea33f91df913d1a>` - deallocate data structures

See the :ref:`examples <doxid-index_dqp_examples>` section for illustrations
of use.

.. include:: irt.rst

.. include:: dqp_functions.rst

available structures
--------------------

.. include :: struct_dqp_control_type.rst

.. include :: struct_dqp_time_type.rst

.. include :: struct_dqp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_dqp_examples:

example calls
-------------

This is an example of how to use the package to solve a given convex quadratic 
program; the code is available in $GALAHAD/src/dqp/Julia/test_dqp.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/dqp/Julia/test_dqp.jl
   :code: julia
