.. _doxid-index_ssls:

GALAHAD SSLS package
====================

.. module:: galahad.ssls

.. include:: ../../Python/ssls_intro.rst

.. include:: ../../Python/ssls_storage.rst

.. toctree::
	:hidden:

	ssls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the ssls package must be 
called in the following order:

* :ref:`ssls_initialize <doxid-galahad__ssls_8h_1a30b1a9463e4abd5cfa0150ffb30569a9>` - provide default control parameters and set up initial data structures

* :ref:`ssls_read_specfile <doxid-galahad__ssls_8h_1abde2e76567a4c8721fe9c2386106e972>` (optional) - override control values by reading replacement values from a file

* :ref:`ssls_import <doxid-galahad__ssls_8h_1ab7cbabccf52f8be7ae417e089eba4b82>` - assemble and analyse the structure of $K$ prior to factorization

* :ref:`ssls_reset_control <doxid-galahad__ssls_8h_1afdfe80ab659c2936d23802b6a6103eb8>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`ssls_factorize_matrix <doxid-galahad__ssls_8h_1a77799da1282c3567b56ae8db42b75f65>` - form and factorize the block matrix $K$ from its components

* :ref:`ssls_solve_system <doxid-galahad__ssls_8h_1a2c3ae7b15fc1c43771d395540c37b9fa>` - solve block linear system of equations involving $K$

* :ref:`ssls_information <doxid-galahad__ssls_8h_1a9f93f5c87ae0088ceb72c4f7e73c9418>` (optional) - recover information about the solution and solution process

* :ref:`ssls_terminate <doxid-galahad__ssls_8h_1a73d7d29d113a62c48cc176146539bca5>` - deallocate data structures

See the :ref:`examples <doxid-index_ssls_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: ssls_functions.rst

available structures
--------------------

.. include :: struct_ssls_control_type.rst

.. include :: struct_ssls_time_type.rst

.. include :: struct_ssls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_ssls_examples:

example calls
-------------

This is an example of how to use the package to solve a block system of 
linear equations; the code is available in $GALAHAD/src/ssls/C/sslst.c .
A variety of supported matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/ssls/C/sslst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/ssls/C/sslstf.c .

.. include :: ../../../src/ssls/C/sslstf.c
   :code: C
