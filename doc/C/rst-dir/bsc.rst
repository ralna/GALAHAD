.. _doxid-index_bsc:

GALAHAD BSC package
===================

.. module:: galahad.bsc

.. include:: ../../Python/bsc_intro.rst

.. include:: ../../Python/bsc_storage.rst

.. toctree::
	:hidden:

	bsc_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bsc package must be 
called in the following order:

* :ref:`bsc_initialize <doxid-galahad__bsc_initialize>` - provide default control parameters and set up initial data structures

* :ref:`bsc_read_specfile <doxid-galahad__bsc_specfile>` (optional) - override control values by reading replacement values from a file

* :ref:`bsc_import <doxid-galahad__bsc_import>` - set up matrix data structures for $A$ and $S$.

* :ref:`bsc_reset_control <doxid-galahad__bsc_reset_control>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`bsc_form_s <doxid-galahad__bsc_form_s>` - form the Schur complement $S$

* :ref:`bsc_information <doxid-galahad__bsc_information>` (optional) - recover information about the process

* :ref:`bsc_terminate <doxid-galahad__bsc_terminate>` - deallocate data structures

See the :ref:`examples <doxid-index_bsc_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: bsc_functions.rst

available structures
--------------------

.. include :: struct_bsc_control_type.rst

.. include :: struct_bsc_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_bsc_examples:

example calls
-------------

This is an example of how to use the package to find the Schur complement 
from given data $A$ and $D$; the code is available in 
$GALAHAD/src/bsc/C/bsct.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/bsc/C/bsct.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/bsc/C/bsctf.c .

.. include :: ../../../src/bsc/C/bsctf.c
   :code: C
