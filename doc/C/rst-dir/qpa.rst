.. _doxid-index_qpa:

GALAHAD QPA package
===================

.. module:: galahad.qpa

.. include:: ../../Python/qpa_intro.rst

.. include:: ../../Python/qpa_storage.rst

.. toctree::
	:hidden:

	qpa_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the qpa package must be 
called in the following order:

* :ref:`qpa_initialize <doxid-galahad__qpa_8h_1afc82144e136ab34fe8a7aea4acd870fc>` - provide default control parameters and set up initial data structures

* :ref:`qpa_read_specfile <doxid-galahad__qpa_8h_1a41a3965eddf14d030a6fa30795149369>` (optional) - override control values by reading replacement values from a file

* :ref:`qpa_import <doxid-galahad__qpa_8h_1a2ecf96b51001b1613ac0637c3bc44824>` - set up problem data structures and fixed values

* :ref:`qpa_reset_control <doxid-galahad__qpa_8h_1a9d60441b2beaddb4c653156592ffc1ea>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`qpa_solve_qp <doxid-galahad__qpa_8h_1af9c60939ef803461d90631dd48cb55d7>` - solve the quadratic program (2)-(4)
  
  * :ref:`qpa_solve_l1qp <doxid-galahad__qpa_8h_1a1a95adb548b743128e0df4ab4e801f19>` - solve the l1 quadratic program (1)
  
  * :ref:`qpa_solve_bcl1qp <doxid-galahad__qpa_8h_1a5b5ef5f0d8134d8f02b1da62a04a3ace>` - solve the bound constrained l1 quadratic program (4)-(5)

* :ref:`qpa_information <doxid-galahad__qpa_8h_1a631cf6ec1a95b27c712ace4fa7dc06f0>` (optional) - recover information about the solution and solution process

* :ref:`qpa_terminate <doxid-galahad__qpa_8h_1a18b73b54796470edc039e3ac85bd30d5>` - deallocate data structures

See the :ref:`examples <doxid-index_qpa_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: qpa_functions.rst

available structures
--------------------

.. include :: struct_qpa_control_type.rst

.. include :: struct_qpa_time_type.rst

.. include :: struct_qpa_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_qpa_examples:

example calls
-------------

This is an example of how to use the package to solve a quadratic program;
the code is available in $GALAHAD/src/qpa/C/qpat.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/qpa/C/qpat.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/qpa/C/qpatf.c .

.. include :: ../../../src/qpa/C/qpatf.c
   :code: C
