.. _doxid-index_bgo:

GALAHAD BGO package
===================

.. module:: galahad.bgo

.. include:: ../../Python/bgo_intro.rst

.. include:: ../../Python/bgo_storage.rst

.. toctree::
	:hidden:

	bgo_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bgo package must be 
called in the following order:

* :ref:`bgo_initialize <doxid-galahad__bgo_8h_1a5d2b6e10b7c04279d6037e9abd32e19f>` - provide default control parameters and set up initial data structures

* :ref:`bgo_read_specfile <doxid-galahad__bgo_8h_1a7a9e6738996809b7fd96a6e9bee6e467>` (optional) - override control values by reading replacement values from a file

* :ref:`bgo_import <doxid-galahad__bgo_8h_1a5902cb8c7c213954de6b963a507f3a4b>` - set up problem data structures and fixed values

* :ref:`bgo_reset_control <doxid-galahad__bgo_8h_1acd46c656b1551f4659d725e65b70e1e6>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`bgo_solve_with_mat <doxid-galahad__bgo_8h_1ab9193a994bd19d94aa97156e83345bd4>` - solve using function calls to evaluate function, gradient and Hessian values
  
  * :ref:`bgo_solve_without_mat <doxid-galahad__bgo_8h_1aeaa490762fe0950e577509ade6ae36d5>` - solve using function calls to evaluate function and gradient values and Hessian-vector products
  
  * :ref:`bgo_solve_reverse_with_mat <doxid-galahad__bgo_8h_1af99998a6921ff67b79e6558fb2a27f2f>` - solve returning to the calling program to obtain function, gradient and Hessian values, or
  
  * :ref:`bgo_solve_reverse_without_mat <doxid-galahad__bgo_8h_1a84e69267132736f46cb7b5970a24b772>` - solve returning to the calling prorgram to obtain function and gradient values and Hessian-vector products

* :ref:`bgo_information <doxid-galahad__bgo_8h_1a96c2a39622f5c497a4286f5e8ebc4ddc>` (optional) - recover information about the solution and solution process

* :ref:`bgo_terminate <doxid-galahad__bgo_8h_1ae41275e1234f0e01ff2aae00746d94d6>` - deallocate data structures

See the :ref:`examples <doxid-index_bgo_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: bgo_functions.rst

available structures
--------------------

.. include :: struct_bgo_control_type.rst

.. include :: struct_bgo_time_type.rst

.. include :: struct_bgo_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_bgo_examples:

example calls
-------------

This is an example of how to use the package to minimize a multi-dimensional
objective within a box; the code is available in $GALAHAD/src/bgo/C/bgot.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/bgo/C/bgot.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/bgo/C/bgotf.c .

.. include :: ../../../src/bgo/C/bgotf.c
   :code: C
