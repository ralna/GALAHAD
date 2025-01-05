.. _doxid-index_trs:

GALAHAD TRS package
===================

.. module:: galahad.trs

.. include:: ../../Python/trs_intro.rst

.. include:: ahm_storage.rst

.. toctree::
	:hidden:

	trs_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the trs package must be
called in the following order:

* :ref:`trs_initialize <doxid-galahad__trs_8h_1acb066d992c4ec394402bc7b7317e1163>` - provide default control parameters and set up initial data structures

* :ref:`trs_read_specfile <doxid-galahad__trs_8h_1adc7c56e7be2f7cc9d32921582d379b13>` (optional) - override control values by reading replacement values from a file

* :ref:`trs_import <doxid-galahad__trs_8h_1a4becded30e9b95fe7028b7799292c0af>` - set up problem data structures and fixed values

* :ref:`trs_import_m <doxid-galahad__trs_8h_1a427420b6025d522bb7b3c652e8c2be48>` - (optional) set up problem data structures and fixed values for the scaling matrix $M$, if any

* :ref:`trs_import_a <doxid-galahad__trs_8h_1ad726ff8f6c25c4384d2b952e8fab4409>` - (optional) set up problem data structures and fixed values for the constraint matrix $A$, if any

* :ref:`trs_reset_control <doxid-galahad__trs_8h_1aae677e64bacb35354f49326815b694c3>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`trs_solve_problem <doxid-galahad__trs_8h_1aadb8a751c29efcef663bf9560a1f9a8e>` - solve the trust-region problem

* :ref:`trs_information <doxid-galahad__trs_8h_1a3dda24010e564e2d6536cc7ea518451e>` (optional) - recover information about the solution and solution process

* :ref:`trs_terminate <doxid-galahad__trs_8h_1ab5cf0077db0631814fdd03599a585376>` - deallocate data structures

See the :ref:`examples <doxid-index_trs_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: trs_functions.rst

available structures
--------------------

.. include :: struct_trs_control_type.rst

.. include :: struct_trs_time_type.rst

.. include :: struct_trs_history_type.rst

.. include :: struct_trs_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_trs_examples:

example calls
-------------

This is an example of how to use the package to solve a trust-region subproblem;
the code is available in $GALAHAD/src/trs/Julia/test_trs.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/trs/Julia/test_trs.jl
   :code: julia
