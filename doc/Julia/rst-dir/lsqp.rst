.. _doxid-index_lsqp:

GALAHAD LSQP package
====================

.. module:: galahad.lsqp

.. include:: ../../Python/lsqp_intro.rst

.. include:: a_storage.rst

.. toctree::
	:hidden:

	lsqp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the lsqp package must be
called in the following order:

* :ref:`lsqp_initialize <doxid-galahad__lsqp_8h_1aac395e385cae77c266ec108f21e9e8f9>` - provide default control parameters and set up initial data structures

* :ref:`lsqp_read_specfile <doxid-galahad__lsqp_8h_1a830242147779223fa2dbed69c2c0c200>` (optional) - override control values by reading replacement values from a file

* :ref:`lsqp_import <doxid-galahad__lsqp_8h_1a5b6f76c31025aa6794e81715f0362e70>` - set up problem data structures and fixed values

* :ref:`lsqp_reset_control <doxid-galahad__lsqp_8h_1a3dc0d9ed7fad6f3ea575e1a53c06c35e>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`lsqp_solve_qp <doxid-galahad__lsqp_8h_1a44019540a174679eda6d46b1ddae89f8>` - solve the quadratic program

* :ref:`lsqp_information <doxid-galahad__lsqp_8h_1aa86d9f9bfe75370d90a0be244a9a23ce>` (optional) - recover information about the solution and solution process

* :ref:`lsqp_terminate <doxid-galahad__lsqp_8h_1a7a9c9d7589c1acb11f0e2a579b1d8053>` - deallocate data structures

See the :ref:`examples <doxid-index_lsqp_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: lsqp_functions.rst

available structures
--------------------

.. include :: struct_lsqp_control_type.rst

.. include :: struct_lsqp_time_type.rst

.. include :: struct_lsqp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lsqp_examples:

example calls
-------------

This is an example of how to use the package to solve a separable quadratic
program; the code is available in $GALAHAD/src/lsqp/Julia/test_lsqp.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/lsqp/Julia/test_lsqp.jl
   :code: julia
