.. _doxid-index_presolve:

GALAHAD PRESOLVE package
========================

.. module:: galahad.presolve

.. include:: presolve_intro.rst

.. include:: ah_storage.rst

.. toctree::
	:hidden:

	presolve_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the presolve package must be 
called in the following order:

* :ref:`presolve_initialize <doxid-galahad__presolve_8h_1a30348a4e0a189046f55d995941693ed9>` - provide default control parameters and set up initial data structures

* :ref:`presolve_read_specfile <doxid-galahad__presolve_8h_1a78f57f6dd2885f41e9b79cc784ff673f>` (optional) - override control values by reading replacement values from a file

* :ref:`presolve_import_problem <doxid-galahad__presolve_8h_1aca96df1bce848a32af9f599a11c4c991>` - import the problem data and report the dimensions of the transformed problem

* :ref:`presolve_transform_problem <doxid-galahad__presolve_8h_1af6da8ac04a1d4fdfd1b91cd8868791a1>` - apply the presolve algorithm to transform the data

* :ref:`presolve_restore_solution <doxid-galahad__presolve_8h_1acf572e4805407de63003cd712f0fc495>` - restore the solution from that of the transformed problem

* :ref:`presolve_information <doxid-galahad__presolve_8h_1adc22ebe32d1361b83889645ff473ca9b>` (optional) - recover information about the solution and solution process

* :ref:`presolve_terminate <doxid-galahad__presolve_8h_1abe2d3138390135885716064c3befb36b>` - deallocate data structures

See the :ref:`examples <doxid-index_presolve_examples>` section for 
illustrations of use.

.. include:: irt.rst

.. include:: presolve_functions.rst

available structures
--------------------

.. include :: struct_presolve_control_type.rst

.. include :: struct_presolve_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_presolve_examples:

example calls
-------------

This is an example of how to use the package to presolve a quadratic program;
the code is available in $GALAHAD/src/presolve/Julia/test_presolve.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/presolve/Julia/test_presolve.jl
   :code: julia
