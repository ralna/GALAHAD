.. _doxid-index_expo:

GALAHAD EXPO package
====================

.. module:: galahad.expo

.. include:: ../../Python/expo_intro.rst

.. include:: ../../Python/expo_storage.rst

.. toctree::
	:hidden:

	expo_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the expo package must be 
called in the following order:

To solve a given problem, functions from the expo package must be called in the following order:

* :ref:`expo_initialize <doxid-galahad__expo_8h_1aa344bb15b74ab3b3ee6afb2de072b19f>` - provide default control parameters and set up initial data structures

* :ref:`expo_read_specfile <doxid-galahad__expo_8h_1adf9db7eff2fce137ae2abd2e013c47b3>` (optional) - override control values by reading replacement values from a file

* :ref:`expo_import <doxid-galahad__expo_8h_1a3f0eb83fd31ee4108156f2e84176389d>` - set up problem data structures and fixed values

* :ref:`expo_reset_control <doxid-galahad__expo_8h_1a07f0857c9923ad0f92d51ed00833afda>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`expo_solve_hessian_direct <doxid-galahad__expo_8h_1ae923c2e6afabb3563fe0998d45b715c4>` - solve the problem using function calls to evaluate function and derivative values

* :ref:`expo_information <doxid-galahad__expo_8h_1a765da96b0a1f3d07dab53cc3400c22d8>` (optional) - recover information about the solution and solution process

* :ref:`expo_terminate <doxid-galahad__expo_8h_1a7babe9112dfad1eb7b57b70135704ab0>` - deallocate data structures

See the :ref:`examples <doxid-index_expo_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: expo_functions.rst

available structures
--------------------

.. include :: struct_expo_control_type.rst

.. include :: struct_expo_time_type.rst

.. include :: struct_expo_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_expo_examples:

example calls
-------------

This is an example of how to use the package to solve a nonlinear least-squares
problem; the code is available in $GALAHAD/src/expo/C/expot.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/forthcoming/expo/C/expot.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/expo/C/expotf.c .

.. include :: ../../../src/forthcoming/expo/C/expotf.c
   :code: C
