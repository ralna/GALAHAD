.. _doxid-index_nodend:

GALAHAD NODEND package
======================

.. module:: galahad.nodend

.. include:: ../../Python/nodend_intro.rst

.. include:: ../../Python/nodend_storage.rst

.. toctree::
	:hidden:

	nodend_functions.rst

introduction to function calls
------------------------------

To find the required permutation, functions from the nodend package must be 
called in the following order:

* :ref:`nodend_initialize <doxid-galahad__nodend_8h_initialize>` - provide default control parameters and set up initial data structures

* :ref:`nodend_read_specfile <doxid-galahad__nodend_8h_read_specfile>` (optional) - override control values by reading replacement values from a file

* :ref:`nodend_import <doxid-galahad__nodend_8h_order>` find a row/colum permutation for sparse Cholesky-like factorization
  
* :ref:`nodend_information <doxid-galahad__nodend_8h_information>` (optional) - recover information about the solution and solution process

See the :ref:`examples <doxid-index_nodend_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: nodend_functions.rst

available structures
--------------------

.. include :: struct_nodend_control_type.rst

.. include :: struct_nodend_time_type.rst

.. include :: struct_nodend_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_nodend_examples:

example calls
-------------

This is an example of how to use the package to find a suitable row/column
permutation of a given matrix; the code is available in 
$GALAHAD/src/nodend/C/nodendt.c .

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/nodend/C/nodendt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/nodend/C/nodendtf.c .

.. include :: ../../../src/nodend/C/nodendtf.c
   :code: C
