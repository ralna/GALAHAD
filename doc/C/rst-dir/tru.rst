.. _doxid-index_tru:

GALAHAD TRU package
===================

.. module:: galahad.tru

.. include:: ../../Python/tru_intro.rst

.. include:: ../../Python/tru_storage.rst

.. toctree::
	:hidden:

	tru_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the tru package must be 
called in the following order:

* :ref:`tru_initialize <doxid-galahad__tru_8h_1af7def0657e11fee556d3006fb64fe267>` - provide default control parameters and set up initial data structures

* :ref:`tru_read_specfile <doxid-galahad__tru_8h_1a870a28132f2747e12d7c93c9ce6ffd01>` (optional) - override control values by reading replacement values from a file

* :ref:`tru_import <doxid-galahad__tru_8h_1a8175a402569a69faa351e2dcd3c48b94>` - set up problem data structures and fixed values

* :ref:`tru_reset_control <doxid-galahad__tru_8h_1a97ce8d0956fdb4165504433a93714495>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`tru_solve_with_mat <doxid-galahad__tru_8h_1a638a31d7027eaf1ae39aa7278e7c5c5a>` - solve using function calls to evaluate function, gradient and Hessian values
  
  * :ref:`tru_solve_without_mat <doxid-galahad__tru_8h_1aaa508227d17d8da723bb0401023acd96>` - solve using function calls to evaluate function and gradient values and Hessian-vector products
  
  * :ref:`tru_solve_reverse_with_mat <doxid-galahad__tru_8h_1a804863856294e362b724fca8953300d5>` - solve returning to the calling program to obtain function, gradient and Hessian values, or
  
  * :ref:`tru_solve_reverse_without_mat <doxid-galahad__tru_8h_1a97252b83eaab0b4d5d3ac53e6b317206>` - solve returning to the calling prorgram to obtain function and gradient values and Hessian-vector products

* :ref:`tru_information <doxid-galahad__tru_8h_1a7c756ce759b44ddbd1ffac77bf497e5a>` (optional) - recover information about the solution and solution process

* :ref:`tru_terminate <doxid-galahad__tru_8h_1aa38f8880b4f63e610ae1f269353ac46e>` - deallocate data structures

See the :ref:`examples <doxid-index_tru_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: tru_functions.rst

available structures
--------------------

.. include :: struct_tru_control_type.rst

.. include :: struct_tru_time_type.rst

.. include :: struct_tru_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_tru_examples:

example calls
-------------

This is an example of how to use the package to minimize a multi-dimenstional
objective function; the code is available in $GALAHAD/src/tru/C/trut.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/tru/C/trut.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/tru/C/trutf.c .

.. include :: ../../../src/tru/C/trutf.c
   :code: C
