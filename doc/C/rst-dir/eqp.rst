.. _doxid-index_eqp:

GALAHAD EQP package
===================

.. module:: galahad.eqp

.. include:: ../../Python/eqp_intro.rst

.. include:: ../../Python/eqp_storage.rst

.. toctree::
	:hidden:

	eqp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the eqp package must be 
called in the following order:

* :ref:`eqp_initialize <doxid-galahad__eqp_8h_1a0d6a00469fba32b588f74c05a386626d>` - provide default control parameters and set up initial data structures

* :ref:`eqp_read_specfile <doxid-galahad__eqp_8h_1a24889f7ee5c51b3c76daab94687faed9>` (optional) - override control values by reading replacement values from a file

* :ref:`eqp_import <doxid-galahad__eqp_8h_1adbadbd11a40aedbbb705334023406de1>` - set up problem data structures and fixed values

* :ref:`eqp_reset_control <doxid-galahad__eqp_8h_1a1b8074313bdc2176203d0c0e9ea87c24>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`eqp_solve_qp <doxid-galahad__eqp_8h_1abf97ed5ee46d2b3fd9a6f75023a95c3d>` - solve the quadratic program
  
  * :ref:`eqp_solve_sldqp <doxid-galahad__eqp_8h_1aaadb310c329e4857b3ad373bcee69e6f>` - solve the shifted least-distance problem

* :ref:`eqp_resolve_qp <doxid-galahad__eqp_8h_1abde96724e9b4b13c5cce1aae5cf93d8f>` (optional) - resolve the problem with the same Hessian and Jacobian, but different $g$, $f$ and/or $c$

* :ref:`eqp_information <doxid-galahad__eqp_8h_1aba10933446f1856fc435ecfbd98371d6>` (optional) - recover information about the solution and solution process

* :ref:`eqp_terminate <doxid-galahad__eqp_8h_1a237e3d3e43c6819205b3f303418324e0>` - deallocate data structures

See the :ref:`examples <doxid-index_eqp_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: eqp_functions.rst

available structures
--------------------

.. include :: struct_eqp_control_type.rst

.. include :: struct_eqp_time_type.rst

.. include :: struct_eqp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_eqp_examples:

example calls
-------------

This is an example of how to use the package to solve an equality-constrained
quadratic program; the code is available in $GALAHAD/src/eqp/C/eqpt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/eqp/C/eqpt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/eqp/C/eqptf.c .

.. include :: ../../../src/eqp/C/eqptf.c
   :code: C
