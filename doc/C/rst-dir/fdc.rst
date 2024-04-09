.. _doxid-index_fdc:

GALAHAD FDC package
===================

.. module:: galahad.fdc

.. include:: ../../Python/fdc_intro.rst

.. include:: ../../Python/fdc_storage.rst

.. toctree::
	:hidden:

	fdc_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the fdc package must be 
called in the following order:

* :ref:`fdc_initialize <doxid-galahad__fdc_8h_1a09ed47873fc4b54eac5b10958939459b>` - provide default control parameters and set up initial data structures

* :ref:`fdc_read_specfile <doxid-galahad__fdc_8h_1aa5e20e6a3ed015cdd927c1bfc7f00a2a>` (optional) - override control values by reading replacement values from a file

* :ref:`fdc_find_dependent_rows <doxid-galahad__fdc_8h_1a37ea723b9a1b8799e7971344858d020a>` - find the number of dependent rows and, if there are any, whether the constraints are independent

* :ref:`fdc_terminate <doxid-galahad__fdc_8h_1a9c0167379258891dee32b35e0529b9f9>` - deallocate data structures

See the :ref:`examples <doxid-index_fdc_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: fdc_functions.rst

available structures
--------------------

.. include :: struct_fdc_control_type.rst

.. include :: struct_fdc_time_type.rst

.. include :: struct_fdc_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_fdc_examples:

example calls
-------------

This is an example of how to use the package to find a subset of independent 
linear constraints; the code is available in $GALAHAD/src/fdc/C/fdct.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/fdc/C/fdct.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/fdc/C/fdctf.c .

.. include :: ../../../src/fdc/C/fdctf.c
   :code: C

