.. _doxid-index_lpa:

GALAHAD LPA package
===================

.. module:: galahad.lpa

.. include:: ../../Python/lpa_intro.rst

.. include:: a_storage.rst

.. toctree::
	:hidden:

	lpa_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the lpa package must be
called in the following order:

To solve a given problem, functions from the lpa package must be called in the following order:

* :ref:`lpa_initialize <doxid-galahad__lpa_8h_1a28046b64b944ea19d21a0c983b980bac>` - provide default control parameters and set up initial data structures

* :ref:`lpa_read_specfile <doxid-galahad__lpa_8h_1a41f85821e1a31f1c2aed14c283bd31e5>` (optional) - override control values by reading replacement values from a file

* :ref:`lpa_import <doxid-galahad__lpa_8h_1a949901759f43b5be6533c7f7508cb6ca>` - set up problem data structures and fixed values

* :ref:`lpa_reset_control <doxid-galahad__lpa_8h_1a7bef5d2b18c73eb3e3471e1e26763627>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`lpa_solve_lp <doxid-galahad__lpa_8h_1a02c01a71baefde6b4a2cf0a819a6bb7c>` - solve the linear program

* :ref:`lpa_information <doxid-galahad__lpa_8h_1a73c17ac59bd8ffe7456fbb1288df5ece>` (optional) - recover information about the solution and solution process

* :ref:`lpa_terminate <doxid-galahad__lpa_8h_1a7c2959b854911544161f0d3699d18d05>` - deallocate data structures

See the :ref:`examples <doxid-index_lpa_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: lpa_functions.rst

available structures
--------------------

.. include :: struct_lpa_control_type.rst

.. include :: struct_lpa_time_type.rst

.. include :: struct_lpa_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lpa_examples:

example calls
-------------

This is an example of how to use the package to solve a linear program;
the code is available in $GALAHAD/src/lpa/Julia/test_lpa.jl .
A variety of supported constraint matrix storage formats are shown.

.. include :: ../../../src/lpa/Julia/test_lpa.jl
   :code: julia
