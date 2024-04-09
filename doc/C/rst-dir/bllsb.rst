.. _doxid-index_bllsb:

GALAHAD BLLSB package
===================

.. module:: galahad.bllsb

.. include:: ../../Python/bllsb_intro.rst

.. include:: ../../Python/bllsb_storage.rst

.. toctree::
	:hidden:

	bllsb_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bllsb package must be called in the following order:

* :ref:`bllsb_initialize <doxid-galahad__bllsb_8h_1a782387ad9cccc5f2e2da9df9016fb923>` - provide default control parameters and set up initial data structures

* :ref:`bllsb_read_specfile <doxid-galahad__bllsb_8h_1ade439e5e06c2852fcb089bb39a667a74>` (optional) - override control values by reading replacement values from a file

* :ref:`bllsb_import <doxid-galahad__bllsb_8h_1a6a2be17b6f871df80bbac93940b83af3>` - set up problem data structures and fixed values

* :ref:`bllsb_reset_control <doxid-galahad__bllsb_8h_1a9f7ccb0cffa909a2be7556edda430190>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`bllsb_solve_blls <doxid-galahad__bllsb_8h_1ac2d720ee7b719bf63c3fa208d37f1bc1>` - solve the bound-constrained linear least-squares problem
  
* :ref:`bllsb_information <doxid-galahad__bllsb_8h_1adfb7589696e4e07fdb65f02bc42c5daf>` (optional) - recover information about the solution and solution process

* :ref:`bllsb_terminate <doxid-galahad__bllsb_8h_1a84e12e9e546f51762d305333dce68e2b>` - deallocate data structures

See the :ref:`examples <doxid-index_bllsb_examples>` section for illustrations 
of use.

callable functions
------------------

.. include:: bllsb_functions.rst

available structures
--------------------

.. include :: struct_bllsb_control_type.rst

.. include :: struct_bllsb_time_type.rst

.. include :: struct_bllsb_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_bllsb_examples:

example calls
-------------

This is an example of how to use the package to solve a given convex quadratic 
program; the code is available in $GALAHAD/src/bllsb/C/bllsbt.c .
A variety of supported design matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/bllsb/C/bllsbt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/bllsb/C/bllsbtf.c .

.. include :: ../../../src/bllsb/C/bllsbtf.c
   :code: C
