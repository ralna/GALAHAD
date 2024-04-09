.. _doxid-index_arc:

GALAHAD ARC package
===================

.. module:: galahad.arc

.. include:: ../../Python/arc_intro.rst

.. include:: ../../Python/arc_storage.rst

.. toctree::
	:hidden:

	arc_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the arc package must be 
called in the following order:

* :ref:`arc_initialize <doxid-galahad__arc_8h_1a54564960edd1c926630be24245773633>` - provide default control parameters and set up initial data structures

* :ref:`arc_read_specfile <doxid-galahad__arc_8h_1ad1eb586a01b707b822210edec1495481>` (optional) - override control values by reading replacement values from a file

* :ref:`arc_import <doxid-galahad__arc_8h_1a4121482e6db477eee55cc2b50bc52835>` - set up problem data structures and fixed values

* :ref:`arc_reset_control <doxid-galahad__arc_8h_1aee92716b81a84655569637e028bc53c8>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`arc_solve_with_mat <doxid-galahad__arc_8h_1aa952762f20eddedff0728c99ef8795b9>` - solve using function calls to evaluate function, gradient and Hessian values
  
  * :ref:`arc_solve_without_mat <doxid-galahad__arc_8h_1aea7f9bc40b893f4df507d807ea8cd670>` - solve using function calls to evaluate function and gradient values and Hessian-vector products
  
  * :ref:`arc_solve_reverse_with_mat <doxid-galahad__arc_8h_1ac47e436d7364399dd7a60efac61ef955>` - solve returning to the calling program to obtain function, gradient and Hessian values, or
  
  * :ref:`arc_solve_reverse_without_mat <doxid-galahad__arc_8h_1a1f098df65cdabfcf80d8e6fb3b1035c5>` - solve returning to the calling prorgram to obtain function and gradient values and Hessian-vector products

* :ref:`arc_information <doxid-galahad__arc_8h_1aec0ce871d494f995e8ad500011a10d56>` (optional) - recover information about the solution and solution process

* :ref:`arc_terminate <doxid-galahad__arc_8h_1a7aa74c74e7ca781532d38d337f0d05eb>` - deallocate data structures

See the :ref:`examples <doxid-index_arc_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: arc_functions.rst

available structures
--------------------

.. include :: struct_arc_control_type.rst

.. include :: struct_arc_time_type.rst

.. include :: struct_arc_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_arc_examples:

example calls
-------------

This is an example of how to use the package to minimize a multi-dimensional
objective function; the code is available in $GALAHAD/src/arc/C/arct.c .
A variety of supported Hessian matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/arc/C/arct.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/arc/C/arctf.c .

.. include :: ../../../src/arc/C/arctf.c
   :code: C
