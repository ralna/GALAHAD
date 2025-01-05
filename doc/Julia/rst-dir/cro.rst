.. _doxid-index_cro:

GALAHAD CRO package
===================

.. module:: galahad.cro

.. include:: ../../Python/cro_intro.rst

.. include:: cro_storage.rst

.. toctree::
	:hidden:

	cro_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the cro package must be
called in the following order:

To solve a given problem, functions from the cro package must be called in the following order:

* :ref:`cro_initialize <doxid-galahad__cro_8h_1aeb10643b5d27efef952b60d9ba0eb206>` - provide default control parameters and set up initial data structures

* :ref:`cro_read_specfile <doxid-galahad__cro_8h_1a55c7770ae26847b5c17055c290a54c2a>` (optional) - override control values by reading replacement values from a file

* :ref:`cro_crossover_solution <doxid-galahad__cro_8h_1a1ab8bdd6e394fe4d89c1c2acba8a5a7b>` - move from a primal-dual soution to a full rank one

* :ref:`cro_terminate <doxid-galahad__cro_8h_1ae0692951f03b0999f73a8f68b7d62212>` - deallocate data structures

See the :ref:`examples <doxid-index_cro_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: cro_functions.rst

available structures
--------------------

.. include :: struct_cro_control_type.rst

.. include :: struct_cro_time_type.rst

.. include :: struct_cro_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_cro_examples:

example calls
-------------

This is an example of how to use the package to crossover from a 
primal-dual QP solution to a basic one;
the code is available in $GALAHAD/src/cro/Julia/test_cro.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/cro/Julia/test_cro.jl
   :code: julia
