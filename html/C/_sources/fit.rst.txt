.. _doxid-index_fit:

GALAHAD FIT package
===================

.. module:: galahad.fit

.. include:: ../../Python/fit_intro.rst

.. include:: ../../Python/fit_storage.rst

.. toctree::
	:hidden:

	fit_functions.rst

callable functions
------------------

.. include:: fit_functions.rst

available structures
--------------------

.. include :: struct_fit_control_type.rst

.. include :: struct_fit_time_type.rst

.. include :: struct_fit_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_fit_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/fit/C/fitt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/fit/C/fitt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/fit/C/fittf.c .

.. include :: ../../../src/fit/C/fittf.c
   :code: C
