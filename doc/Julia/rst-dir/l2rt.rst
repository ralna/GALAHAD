.. _doxid-index_l2rt:

GALAHAD L2RT package
====================

.. module:: galahad.l2rt

.. include:: ../../Python/l2rt_intro.rst

.. toctree::
	:hidden:

	l2rt_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the l2rt package must be 
called in the following order:

* :ref:`l2rt_initialize <doxid-galahad__l2rt_8h_1a0103448a3db662f9c483f9f44a5112bc>` - provide default control parameters and set up initial data structures

* :ref:`l2rt_read_specfile <doxid-galahad__l2rt_8h_1a1b63f8b501208629cceb662b03f35684>` (optional) - override control values by reading replacement values from a file

* :ref:`l2rt_import_control <doxid-galahad__l2rt_8h_1adf880b26c8aea32493857f8576e58ae8>` - import control parameters prior to solution

* :ref:`l2rt_solve_problem <doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754>` - solve the problem by reverse communication, a sequence of calls are made under control of a status parameter, each exit either asks the user to provide additional informaton and to re-enter, or reports that either the solution has been found or that an error has occurred

* :ref:`l2rt_information <doxid-galahad__l2rt_8h_1a4fa18245556cf87b255b2b9ac5748ca9>` (optional) - recover information about the solution and solution process

* :ref:`l2rt_terminate <doxid-galahad__l2rt_8h_1aa9b62de33c3d6c129cca1e90a3d548b7>` - deallocate data structures

See the :ref:`examples <doxid-index_l2rt_examples>` section for 
illustrations of use.

.. include:: irt.rst

.. include:: l2rt_functions.rst

available structures
--------------------

.. include :: struct_l2rt_control_type.rst

.. include :: struct_l2rt_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_l2rt_examples:

example calls
-------------

This is an example of how to use the package to solve a regularized 
least-Euclidean-norm subproblem;
the code is available in $GALAHAD/src/l2rt/Julia/test_l2rt.jl .

.. include :: ../../../src/l2rt/Julia/test_l2rt.jl
   :code: julia
