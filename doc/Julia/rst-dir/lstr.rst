.. _doxid-index_lstr:

GALAHAD LSTR package
====================

.. module:: galahad.lstr

.. include:: ../../Python/lstr_intro.rst

.. toctree::
	:hidden:

	lstr_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the lstr package must be 
called in the following order:

* :ref:`lstr_initialize <doxid-galahad__lstr_8h_1ae423bf7ffc77c89f461448ca1f5c286c>` - provide default control parameters and set up initial data structures

* :ref:`lstr_read_specfile <doxid-galahad__lstr_8h_1a3d3fa989fe4c3b40cd7e296249d2205d>` (optional) - override control values by reading replacement values from a file

* :ref:`lstr_import_control <doxid-galahad__lstr_8h_1a1a8ad63d944dc046fd2040554d6d01e5>` - import control parameters prior to solution

* :ref:`lstr_solve_problem <doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562>` - solve the problem by reverse communication, a sequence of calls are made under control of a status parameter, each exit either asks the user to provide additional informaton and to re-enter, or reports that either the solution has been found or that an error has occurred

* :ref:`lstr_information <doxid-galahad__lstr_8h_1a5929f00ea00af253ede33a6749451481>` (optional) - recover information about the solution and solution process

* :ref:`lstr_terminate <doxid-galahad__lstr_8h_1aa198189942e179e52699e1fedfcdf9d1>` - deallocate data structures

See the :ref:`examples <doxid-index_lstr_examples>` section for 
illustrations of use.

.. include:: irt.rst

.. include:: lstr_functions.rst

available structures
--------------------

.. include :: struct_lstr_control_type.rst

.. include :: struct_lstr_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lstr_examples:

example calls
-------------

This is an example of how to use the package to solve a linear least-squares 
trust-region problem;
the code is available in $GALAHAD/src/lstr/Julia/test_lstr.jl .

.. include :: ../../../src/lstr/Julia/test_lstr.jl
   :code: julia
