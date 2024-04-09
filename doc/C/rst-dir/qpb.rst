.. _doxid-index_qpb:

GALAHAD QPB package
===================

.. module:: galahad.qpb

.. include:: ../../Python/qpb_intro.rst

.. include:: ../../Python/qpb_storage.rst

.. toctree::
	:hidden:

	qpb_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the qpb package must be 
called in the following order:

* :ref:`qpb_initialize <doxid-galahad__qpb_8h_1afdfb45debf15ddcc3d6e0feea4e34784>` - provide default control parameters and set up initial data structures

* :ref:`qpb_read_specfile <doxid-galahad__qpb_8h_1a162514dc80468d390e80c620890c8710>` (optional) - override control values by reading replacement values from a file

* :ref:`qpb_import <doxid-galahad__qpb_8h_1a36559bb33d97b61e94e0c7dfa73b67d8>` - set up problem data structures and fixed values

* :ref:`qpb_reset_control <doxid-galahad__qpb_8h_1a42484e4b1aafb880fcdb215d5683a652>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`qpb_solve_qp <doxid-galahad__qpb_8h_1ac83f8c83e64a7f130fb247f162169472>` - solve the quadratic program

* :ref:`qpb_information <doxid-galahad__qpb_8h_1a2b77d21a6c613462657e8b7f51b6b1d2>` (optional) - recover information about the solution and solution process

* :ref:`qpb_terminate <doxid-galahad__qpb_8h_1a35596f23213b063f1f8abfc6f796bc77>` - deallocate data structures

See the :ref:`examples <doxid-index_qpb_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: qpb_functions.rst

available structures
--------------------

.. include :: struct_qpb_control_type.rst

.. include :: struct_qpb_time_type.rst

.. include :: struct_qpb_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_qpb_examples:

example calls
-------------

This is an example of how to use the package to solve a quadratic program;
the code is available in $GALAHAD/src/qpb/C/qpbt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/qpb/C/qpbt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/qpb/C/qpbtf.c .

.. include :: ../../../src/qpb/C/qpbtf.c
   :code: C
