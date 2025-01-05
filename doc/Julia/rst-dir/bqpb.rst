.. _doxid-index_bqpb:

GALAHAD BQPB package
====================

.. module:: galahad.bqpb

.. include:: ../../Python/bqpb_intro.rst

.. include:: h_storage.rst

.. toctree::
	:hidden:

	bqpb_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bqpb package must be
called in the following order:

* :ref:`bqpb_initialize <doxid-galahad__bqpb_8h_1ad8fc12f75d4b6ca96fd0912785a04b6f>` - provide default control parameters and set up initial data structures

* :ref:`bqpb_read_specfile <doxid-galahad__bqpb_8h_1a4702d5710e0b6dd9e4fd05e30cc1915b>` (optional) - override control values by reading replacement values from a file

* :ref:`bqpb_import <doxid-galahad__bqpb_8h_1a489747f9b6b3edd736b318add2e6e96d>` - set up problem data structures and fixed values

* :ref:`bqpb_reset_control <doxid-galahad__bqpb_8h_1a28853e7625bc052a96d6189ac3c8bd04>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of

  * :ref:`bqpb_solve_qp <doxid-galahad__bqpb_8h_1afdd78a23df912116a044a3cd87b082c1>` - solve the bound-constrained quadratic program

  * :ref:`bqpb_solve_sldqp <doxid-galahad__bqpb_8h_1aa1378c5f67c67450b853cd33f978e0d7>` - solve the bound-constrained shifted least-distance problem

* :ref:`bqpb_information <doxid-galahad__bqpb_8h_1a01c7e22011ff22e8084be1e8a26d84c6>` (optional) - recover information about the solution and solution process

* :ref:`bqpb_terminate <doxid-galahad__bqpb_8h_1a6a2b870d2c3d4907b4551e7abc700893>` - deallocate data structures

See the :ref:`examples <doxid-index_bqpb_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: bqpb_functions.rst

available structures
--------------------

.. include :: struct_bqpb_control_type.rst

.. include :: struct_bqpb_time_type.rst

.. include :: struct_bqpb_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_bqpb_examples:

example calls
-------------

This is an example of how to use the package to solve a bound-constrained QP;
the code is available in $GALAHAD/src/bqpb/Julia/test_bqpb.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/bqpb/Julia/test_bqpb.jl
   :code: julia
