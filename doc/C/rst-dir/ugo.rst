.. _doxid-index_ugo:

GALAHAD UGO package
===================

.. module:: galahad.ugo

.. include:: ../../Python/ugo_intro.rst

.. toctree::
	:hidden:

	ugo_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the ugo package must be 
called in the following order:

* :ref:`ugo_initialize <doxid-galahad__ugo_8h_1a172105bd528410f7c7e2fd77899ebc78>` - provide default control parameters and set up initial data structures

* :ref:`ugo_read_specfile <doxid-galahad__ugo_8h_1a6819d58a728f3bf97232ed719e72fb91>` (optional) - override control values by reading replacement values from a file

* :ref:`ugo_import <doxid-galahad__ugo_8h_1a8bcbdf9ef1229535b77d9991eb543dcb>` - set up problem data structures and fixed values

* :ref:`ugo_reset_control <doxid-galahad__ugo_8h_1a51fa6faacfb75c3dcad44befd2e6cb40>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`ugo_solve_direct <doxid-galahad__ugo_8h_1aa5b2949ab17e25a0a0c24f38c0d61a1a>` - solve using function calls to evaluate function and derivative values, or
  
  * :ref:`ugo_solve_reverse <doxid-galahad__ugo_8h_1a0b8f123f8e67bb0cb8a27c5ce87c824c>` - solve returning to the calling program to obtain function and derivative values

* :ref:`ugo_information <doxid-galahad__ugo_8h_1a8e1db35daea3247b2cc9eb8607d0abee>` (optional) - recover information about the solution and solution process

* :ref:`ugo_terminate <doxid-galahad__ugo_8h_1ad9485926c547bb783aea3ee1adb3b084>` - deallocate data structures

See the :ref:`examples <doxid-index_ugo_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: ugo_functions.rst

available structures
--------------------

.. include :: struct_ugo_control_type.rst

.. include :: struct_ugo_time_type.rst

.. include :: struct_ugo_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_ugo_examples:

example calls
-------------

This is an example of how to use the package to minimize a univariate function;
the code is available in $GALAHAD/src/ugo/C/ugot.c .

The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/ugo/C/ugot.c
   :code: C
