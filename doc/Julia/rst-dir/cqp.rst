.. _doxid-index_cqp:

GALAHAD CQP package
===================

.. module:: galahad.cqp

.. include:: ../../Python/cqp_intro.rst

.. include:: ah_storage.rst

.. toctree::
	:hidden:

	cqp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the cqp package must be called in the following order:

* :ref:`cqp_initialize <doxid-galahad__cqp_8h_1a782387ad9cccc5f2e2da9df9016fb923>` - provide default control parameters and set up initial data structures

* :ref:`cqp_read_specfile <doxid-galahad__cqp_8h_1ade439e5e06c2852fcb089bb39a667a74>` (optional) - override control values by reading replacement values from a file

* :ref:`cqp_import <doxid-galahad__cqp_8h_1a6a2be17b6f871df80bbac93940b83af3>` - set up problem data structures and fixed values

* :ref:`cqp_reset_control <doxid-galahad__cqp_8h_1a9f7ccb0cffa909a2be7556edda430190>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of

  * :ref:`cqp_solve_qp <doxid-galahad__cqp_8h_1ac2d720ee7b719bf63c3fa208d37f1bc1>` - solve the quadratic program

  * :ref:`cqp_solve_sldqp <doxid-galahad__cqp_8h_1a311864de053b1cbbf78b1cbab052e56e>` - solve the shifted least-distance problem

* :ref:`cqp_information <doxid-galahad__cqp_8h_1adfb7589696e4e07fdb65f02bc42c5daf>` (optional) - recover information about the solution and solution process

* :ref:`cqp_terminate <doxid-galahad__cqp_8h_1a84e12e9e546f51762d305333dce68e2b>` - deallocate data structures

See the :ref:`examples <doxid-index_cqp_examples>` section for illustrations
of use.

.. include:: irt.rst

.. include:: cqp_functions.rst

available structures
--------------------

.. include :: struct_cqp_control_type.rst

.. include :: struct_cqp_time_type.rst

.. include :: struct_cqp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_cqp_examples:

example calls
-------------

This is an example of how to use the package to solve a given convex quadratic 
program; the code is available in $GALAHAD/src/cqp/Julia/test_cqp.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/cqp/Julia/test_cqp.jl
   :code: julia
