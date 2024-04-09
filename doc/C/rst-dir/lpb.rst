.. _doxid-index_lpb:

GALAHAD LPB package
===================

.. module:: galahad.lpb

.. include:: ../../Python/lpb_intro.rst

.. include:: ../../Python/lpb_storage.rst

.. toctree::
	:hidden:

	lpb_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the lpb package must be 
called in the following order:

* :ref:`lpb_initialize <doxid-galahad__lpb_8h_1a63dd5d968d870274e0abc9c3e1e553f6>` - provide default control parameters and set up initial data structures

* :ref:`lpb_read_specfile <doxid-galahad__lpb_8h_1ad3cf4da5c65e4b31d2d4ff45a392c567>` (optional) - override control values by reading replacement values from a file

* :ref:`lpb_import <doxid-galahad__lpb_8h_1ac3308e860ab39acf9d7f293f75d80fbd>` - set up problem data structures and fixed values

* :ref:`lpb_reset_control <doxid-galahad__lpb_8h_1aac79b2577895e28d4a92deb9f3bd24a6>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`lpb_solve_lp <doxid-galahad__lpb_8h_1a3fecba0e7ec404089d904a5623e0e83e>` - solve the linear program

* :ref:`lpb_information <doxid-galahad__lpb_8h_1ad3b3173cbeb7a9b01995d678324cbe4e>` (optional) - recover information about the solution and solution process

* :ref:`lpb_terminate <doxid-galahad__lpb_8h_1ac7258f2afb0b15c191838ecfa377d264>` - deallocate data structures


See the :ref:`examples <doxid-index_lpb_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: lpb_functions.rst

available structures
--------------------

.. include :: struct_lpb_control_type.rst

.. include :: struct_lpb_time_type.rst

.. include :: struct_lpb_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lpb_examples:

example calls
-------------

This is an example of how to use the package to solve a linear program;
the code is available in $GALAHAD/src/lpb/C/lpbt.c .
A variety of supported constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/lpb/C/lpbt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/lpb/C/lpbtf.c .

.. include :: ../../../src/lpb/C/lpbtf.c
   :code: C
