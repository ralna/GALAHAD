.. _doxid-index_glrt:

GALAHAD GLRT package
====================

.. module:: galahad.glrt

.. include:: ../../Python/glrt_intro.rst

.. toctree::
	:hidden:

	glrt_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the glrt package must be 
called in the following order:

* :ref:`glrt_initialize <doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32>` - provide default control parameters and set up initial data structures

* :ref:`glrt_read_specfile <doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725>` (optional) - override control values by reading replacement values from a file

* :ref:`glrt_import_control <doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1>` - import control parameters prior to solution

* :ref:`glrt_solve_problem <doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>` - solve the problem by reverse communication, a sequence of calls are made under control of a status parameter, each exit either asks the user to provide additional informaton and to re-enter, or reports that either the solution has been found or that an error has occurred

* :ref:`glrt_information <doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d>` (optional) - recover information about the solution and solution process

* :ref:`glrt_terminate <doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812>` - deallocate data structures

See the :ref:`examples <doxid-index_glrt_examples>` section for 
illustrations of use.

.. include:: irt.rst

.. include:: glrt_functions.rst

available structures
--------------------

.. include :: struct_glrt_control_type.rst

.. include :: struct_glrt_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_glrt_examples:

example calls
-------------

This is an example of how to use the package to solve a regularization 
subproblem; the code is available in $GALAHAD/src/glrt/Julia/test_glrt.jl .

.. include :: ../../../src/glrt/Julia/test_glrt.jl
   :code: julia
