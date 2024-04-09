.. _doxid-index_gltr:

GALAHAD GLTR package
====================

.. module:: galahad.gltr

.. include:: ../../Python/gltr_intro.rst

.. toctree::
	:hidden:

	gltr_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the gltr package must be 
called in the following order:

* :ref:`gltr_initialize <doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c>` - provide default control parameters and set up initial data structures

* :ref:`gltr_read_specfile <doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31>` (optional) - override control values by reading replacement values from a file

* :ref:`gltr_import_control <doxid-galahad__gltr_8h_1acb8a654fc381e3f231c3d10858f111b3>` - import control parameters prior to solution

* :ref:`gltr_solve_problem <doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18>` - solve the problem by reverse communication, a sequence of calls are made under control of a status parameter, each exit either asks the user to provide additional informaton and to re-enter, or reports that either the solution has been found or that an error has occurred

* :ref:`gltr_information <doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb>` (optional) - recover information about the solution and solution process

* :ref:`gltr_terminate <doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631>` - deallocate data structures

See the :ref:`examples <doxid-index_gltr_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: gltr_functions.rst

available structures
--------------------

.. include :: struct_gltr_control_type.rst

.. include :: struct_gltr_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_gltr_examples:

example calls
-------------

This is an example of how to use the package to solve a trust-region subproblem;
the code is available in $GALAHAD/src/gltr/C/gltrt.c .

The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/gltr/C/gltrt.c
   :code: C
