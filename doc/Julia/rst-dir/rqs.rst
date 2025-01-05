.. _doxid-index_rqs:

GALAHAD RQS package
===================

.. module:: galahad.rqs

.. include:: ../../Python/rqs_intro.rst

.. include:: ahm_storage.rst

.. toctree::
	:hidden:

	rqs_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the rqs package must be
called in the following order:

* :ref:`rqs_initialize <doxid-galahad__rqs_8h_1aeb8c3e1a278c83094aaaf185e9833fac>` - provide default control parameters and set up initial data structures

* :ref:`rqs_read_specfile <doxid-galahad__rqs_8h_1a1f6f3841ad5f7952dbc04a7cb19dd0e7>` (optional) - override control values by reading replacement values from a file

* :ref:`rqs_import <doxid-galahad__rqs_8h_1af815172e77293aa2a7c9dbcac2379f50>` - set up problem data structures and fixed values

* :ref:`rqs_import_m <doxid-galahad__rqs_8h_1af0351d4956431c86e229f905041c222b>` - (optional) set up problem data structures and fixed values for the scaling matrix $M$, if any

* :ref:`rqs_import_a <doxid-galahad__rqs_8h_1a3d1116ac5c18fe085e902c77ec2776b5>` - (optional) set up problem data structures and fixed values for the constraint matrix $A$, if any

* :ref:`rqs_reset_control <doxid-galahad__rqs_8h_1a86e1c32d2d07facbe602222e199a075f>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`rqs_solve_problem <doxid-galahad__rqs_8h_1a162e2301c9d4bde7d57f5f1e820e2b84>` - solve the regularised quadratic problem

* :ref:`rqs_information <doxid-galahad__rqs_8h_1a586e85ec11c4647346916f49805fcb83>` (optional) - recover information about the solution and solution process

* :ref:`rqs_terminate <doxid-galahad__rqs_8h_1ae1f727eadfaada300dc6a5e268ac2b74>` - deallocate data structures

See the :ref:`examples <doxid-index_rqs_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: rqs_functions.rst

available structures
--------------------

.. include :: struct_rqs_control_type.rst

.. include :: struct_rqs_time_type.rst

.. include :: struct_rqs_history_type.rst

.. include :: struct_rqs_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_rqs_examples:

example calls
-------------

This is an example of how to use the package to solve a regularized quadratic
subproblem; the code is available in $GALAHAD/src/rqs/Julia/test_rqs.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/rqs/Julia/test_rqs.jl
   :code: julia
