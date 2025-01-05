.. _doxid-index_trb:

GALAHAD TRB package
===================

.. module:: galahad.trb

.. include:: ../../Python/trb_intro.rst

.. include:: hx_storage.rst

.. toctree::
	:hidden:

	trb_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the trb package must be
called in the following order:

* :ref:`trb_initialize <doxid-galahad__trb_8h_1a9bffc46178a3e0b7eb2927d1c50440a1>` - provide default control parameters and set up initial data structures

* :ref:`trb_read_specfile <doxid-galahad__trb_8h_1a4eaafdaf5187c8b91c119ce9395469e9>` (optional) - override control values by reading replacement values from a file

* :ref:`trb_import <doxid-galahad__trb_8h_1a13bc38fb28201adb78af7acf910ff0d8>` - set up problem data structures and fixed values

* :ref:`trb_reset_control <doxid-galahad__trb_8h_1a550c3ca1966ea0fa9de84423b8658cd7>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of

  * :ref:`trb_solve_with_mat <doxid-galahad__trb_8h_1a5a58e6c0c022eb451f14c82d653967f7>` - solve using function calls to evaluate function, gradient and Hessian values

  * :ref:`trb_solve_without_mat <doxid-galahad__trb_8h_1a376b81748cec0bf992542d80b8d38f49>` - solve using function calls to evaluate function and gradient values and Hessian-vector products

  * :ref:`trb_solve_reverse_with_mat <doxid-galahad__trb_8h_1a7bb520e36666386824b216a84be08837>` - solve returning to the calling program to obtain function, gradient and Hessian values, or

  * :ref:`trb_solve_reverse_without_mat <doxid-galahad__trb_8h_1a95eac11acf02fe0d6eb4bc39ace5a100>` - solve returning to the calling prorgram to obtain function and gradient values and Hessian-vector products

* :ref:`trb_information <doxid-galahad__trb_8h_1a105f41a31d49c59885c3372090bec776>` (optional) - recover information about the solution and solution process

* :ref:`trb_terminate <doxid-galahad__trb_8h_1a739c7a44ddd0ce7b350cfa6da54948d0>` - deallocate data structures

See the :ref:`examples <doxid-index_trb_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: trb_functions.rst

available structures
--------------------

.. include :: struct_trb_control_type.rst

.. include :: struct_trb_time_type.rst

.. include :: struct_trb_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_trb_examples:

example calls
-------------

This is an example of how to use the package to solve a bound-constrained
multi-dimensional optimization problem; 
the code is available in  $GALAHAD/src/trb/Julia/test_trb.jl .
A variety of supported Hessian matrix storage formats are shown.

.. include :: ../../../src/trb/Julia/test_trb.jl
   :code: julia
