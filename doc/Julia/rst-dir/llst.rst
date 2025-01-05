.. _doxid-index_llst:

GALAHAD LLST package
====================

.. module:: galahad.llst

.. include:: ../../Python/llst_intro.rst

.. include:: as_storage.rst

.. toctree::
	:hidden:

	llst_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the llst package must be
called in the following order:

* :ref:`llst_initialize <doxid-galahad__llst_8h_1a9da7a4daba2ceaf875fbd24fe42fbe1f>` - provide default control parameters and set up initial data structures

* :ref:`llst_read_specfile <doxid-galahad__llst_8h_1a9bcda9a7420b5de742370e1464d5b0c2>` (optional) - override control values by reading replacement values from a file

* :ref:`llst_import <doxid-galahad__llst_8h_1a4ffc854176462b1d6492b55317150236>` - set up problem data structures and fixed values

* :ref:`llst_import_scaling <doxid-galahad__llst_8h_1a42d56aec0cdf37373e5a50b13b4c374f>` (optional) - set up problem data structures for $S$ if required

* :ref:`llst_reset_control <doxid-galahad__llst_8h_1a920e8696eea77dab3348a663a1127b41>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`llst_solve_problem <doxid-galahad__llst_8h_1add75b5481c528cca64abbcdeb3a2af35>` - solve the trust-region problem

* :ref:`llst_information <doxid-galahad__llst_8h_1a88854815d1c936131dcc762c64275d6f>` (optional) - recover information about the solution and solution process

* :ref:`llst_terminate <doxid-galahad__llst_8h_1a3d7693551362082a30094e7dea5a2a66>` - deallocate data structures


See the :ref:`examples <doxid-index_llst_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: llst_functions.rst

available structures
--------------------

.. include :: struct_llst_control_type.rst

.. include :: struct_llst_time_type.rst

.. include :: struct_llst_history_type.rst

.. include :: struct_llst_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_llst_examples:

example calls
-------------

This is an example of how to use the package to solve a linear least-squares
trust-region subproblem;
the code is available in $GALAHAD/src/llst/Julia/test_llst.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/llst/Julia/test_llst.jl
   :code: julia
