.. _doxid-index_blls:

GALAHAD BLLS package
====================

.. module:: galahad.blls

.. include:: ../../Python/blls_intro.rst

.. include:: ao_storage.rst

.. toctree::
	:hidden:

	blls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the blls package must be
called in the following order:

* :ref:`blls_initialize <doxid-galahad__blls_8h_1a12708c98f2473e03cd46f4dcfdb03409>` - provide default control parameters and set up initial data structures

* :ref:`blls_read_specfile <doxid-galahad__blls_8h_1aa24c9c2fdaaaac84df5b98abbf84c859>` (optional) - override control values by reading replacement values from a file

* set up problem data structures and fixed values by caling one of

  * :ref:`blls_import <doxid-galahad__blls_8h_1afacd84f0b7592f4532cf7b77d278282f>` - in the case that $A_o$ is explicitly available

  * :ref:`blls_import_without_a <doxid-galahad__blls_8h_1a419f9b0769b4389beffbbc5f7d0fd58c>` - in the case that only the effect of applying $A_o$ and its transpose to a vector is possible

* :ref:`blls_reset_control <doxid-galahad__blls_8h_1a96981ac9a0e3f44b2b38362fc3ab9991>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of

  * :ref:`blls_solve_given_a <doxid-galahad__blls_8h_1acf6d292989a5ac09f7f3e507283fb5bf>` - solve the problem using values of $A_o$

  * :ref:`blls_solve_reverse_a_prod <doxid-galahad__blls_8h_1ac139bc1c65cf12cb532c4ab09f3af9d0>` - solve the problem by returning to the caller for products of $A_o$ and its transpose with specified vectors

* :ref:`blls_information <doxid-galahad__blls_8h_1a457b8ee7c630715bcb43427f254b555f>` (optional) - recover information about the solution and solution process

* :ref:`blls_terminate <doxid-galahad__blls_8h_1ade863ffb6b142bfce669729f56911ac1>` - deallocate data structures

See the :ref:`examples <doxid-index_blls_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: blls_functions.rst

available structures
--------------------

.. include :: struct_blls_control_type.rst

.. include :: struct_blls_time_type.rst

.. include :: struct_blls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_blls_examples:

example calls
-------------

This is an example of how to use the package to solve a bound-constrained 
linear least-squares problem; the code is available in 
$GALAHAD/src/blls/Julia/test_blls.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/blls/Julia/test_blls.jl
   :code: julia
