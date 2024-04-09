.. _doxid-index_llsr:

GALAHAD LLSR package
====================

.. module:: galahad.llsr

.. include:: ../../Python/llsr_intro.rst

.. include:: ../../Python/llsr_storage.rst

.. toctree::
	:hidden:

	llsr_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the llsr package must be 
called in the following order:

* :ref:`llsr_initialize <doxid-galahad__llsr_8h_1a926f9824ab2e2bc71450a4d0b483879d>` - provide default control parameters and set up initial data structures

* :ref:`llsr_read_specfile <doxid-galahad__llsr_8h_1a01bf988188420284ac4c125fef947efb>` (optional) - override control values by reading replacement values from a file

* :ref:`llsr_import <doxid-galahad__llsr_8h_1ac2dd0bee9270e6295c63a5365186070f>` - set up problem data structures and fixed values

* :ref:`llsr_import_scaling <doxid-galahad__llsr_8h_1a75f3108d65fc8100776af18f6adf4c2c>` (optional) - set up problem data structures for $S$ if required

* :ref:`llsr_reset_control <doxid-galahad__llsr_8h_1a9a9e3ae8ce66a5b7933b06061208c50c>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`llsr_solve_problem <doxid-galahad__llsr_8h_1aa702f5ffcde083da921921c4e1131d89>` - solve the regularization problem

* :ref:`llsr_information <doxid-galahad__llsr_8h_1a1ba2eb99bc6364f476b85c7f507d43a2>` (optional) - recover information about the solution and solution process

* :ref:`llsr_terminate <doxid-galahad__llsr_8h_1af05d27e76348279a8c9c16298a819609>` - deallocate data structures

See the :ref:`examples <doxid-index_llsr_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: llsr_functions.rst

available structures
--------------------

.. include :: struct_llsr_control_type.rst

.. include :: struct_llsr_time_type.rst

.. include :: struct_llsr_history_type.rst

.. include :: struct_llsr_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_llsr_examples:

example calls
-------------

This is an example of how to use the package to solve a regularized linear
least-squares subproblem;
the code is available in $GALAHAD/src/llsr/C/llsrt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/llsr/C/llsrt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/llsr/C/llsrtf.c .

.. include :: ../../../src/llsr/C/llsrtf.c
   :code: C
