.. _doxid-index_trek:

GALAHAD TREK package
====================

.. module:: galahad.trek

.. include:: ../../Python/trek_intro.rst

.. include:: hs_storage.rst

.. toctree::
	:hidden:

	trek_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the trek package must be
called in the following order:

* :ref:`trek_initialize <doxid-galahad__trek_8h_1acb066d992c4ec394402bc7b7317e1163>` - provide default control parameters and set up initial data structures

* :ref:`trek_read_specfile <doxid-galahad__trek_8h_1adc7c56e7be2f7cc9d32921582d379b13>` (optional) - override control values by reading replacement values from a file

* :ref:`trek_import <doxid-galahad__trek_8h_1a4becded30e9b95fe7028b7799292c0af>` - set up problem data structures and fixed values

* :ref:`trek_import_s <doxid-galahad__trek_8h_1a427420b6025d522bb7b3c652e8c2be48>` - (optional) set up problem data structures and fixed values for the scaling matrix $S$, if any

* :ref:`trek_reset_control <doxid-galahad__trek_8h_1aae677e64bacb35354f49326815b694c3>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`trek_solve_problem <doxid-galahad__trek_8h_1aadb8a751c29efcef663bf9560a1f9a8e>` - solve the trust-region problem

* :ref:`trek_information <doxid-galahad__trek_8h_1a3dda24010e564e2d6536cc7ea518451e>` (optional) - recover information about the solution and solution process

* :ref:`trek_terminate <doxid-galahad__trek_8h_1ab5cf0077db0631814fdd03599a585376>` - deallocate data structures

See the :ref:`examples <doxid-index_trek_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: trek_functions.rst

available structures
--------------------

.. include :: struct_trek_control_type.rst

.. include :: struct_trek_time_type.rst

.. include :: struct_trek_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_trek_examples:

example calls
-------------

This is an example of how to use the package to solve a trust-region subproblem;
the code is available in $GALAHAD/src/trek/Julia/test_trek.jl .
A variety of supported Hessian and scaling matrix storage formats are shown.

.. include :: ../../../src/trek/Julia/test_trek.jl
   :code: julia
