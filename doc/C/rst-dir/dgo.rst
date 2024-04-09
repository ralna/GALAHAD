.. _doxid-index_dgo:

GALAHAD DGO package
===================

.. module:: galahad.dgo

.. include:: ../../Python/dgo_intro.rst

.. include:: ../../Python/dgo_storage.rst

.. toctree::
	:hidden:

	dgo_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the dgo package must be 
called in the following order:

* :ref:`dgo_initialize <doxid-galahad__dgo_8h_1a80425d4671e565a45c13aa026f6897ef>` - provide default control parameters and set up initial data structures

* :ref:`dgo_read_specfile <doxid-galahad__dgo_8h_1ab8ba227e6d624a0197afab9f77bbe66a>` (optional) - override control values by reading replacement values from a file

* :ref:`dgo_import <doxid-galahad__dgo_8h_1ace7cbe696d8be7026753681d9b7cd149>` - set up problem data structures and fixed values

* :ref:`dgo_reset_control <doxid-galahad__dgo_8h_1ab52e88675fc811f7e9bc38148d42e932>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`dgo_solve_with_mat <doxid-galahad__dgo_8h_1a3b573f5a56c7162383a757221a5b7a36>` - solve using function calls to evaluate function, gradient and Hessian values
  
  * :ref:`dgo_solve_without_mat <doxid-galahad__dgo_8h_1a6ea7cfa79c25e784d21e10cc26ed9954>` - solve using function calls to evaluate function and gradient values and Hessian-vector products
  
  * :ref:`dgo_solve_reverse_with_mat <doxid-galahad__dgo_8h_1a02f408b215596c01b0e3836dfa301b9f>` - solve returning to the calling program to obtain function, gradient and Hessian values, or
  
  * :ref:`dgo_solve_reverse_without_mat <doxid-galahad__dgo_8h_1a878a7d98d55794fa38f885a5d76aa4f0>` - solve returning to the calling prorgram to obtain function and gradient values and Hessian-vector products

* :ref:`dgo_information <doxid-galahad__dgo_8h_1aea0c208de08f507be7a31fe3ab7d3b91>` (optional) - recover information about the solution and solution process

* :ref:`dgo_terminate <doxid-galahad__dgo_8h_1ad12337a0c7ad3ac74e7f8c0783fbbfab>` - deallocate data structures

See the :ref:`examples <doxid-index_dgo_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: dgo_functions.rst

available structures
--------------------

.. include :: struct_dgo_control_type.rst

.. include :: struct_dgo_time_type.rst

.. include :: struct_dgo_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_dgo_examples:

example calls
-------------

This is an example of how to use the package to minimize a multi-dimensional
objective; the code is available in $GALAHAD/src/dgo/C/dgot.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/dgo/C/dgot.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/dgo/C/dgotf.c .

.. include :: ../../../src/dgo/C/dgotf.c
   :code: C
