.. _doxid-index_convert:

GALAHAD CONVERT package
=======================

.. module:: galahad.convert

.. include:: ../../Python/convert_intro.rst

.. include:: ../../Python/convert_storage.rst

.. toctree::
	:hidden:

	convert_functions.rst

callable functions
------------------

.. include:: convert_functions.rst

available structures
--------------------

.. include :: struct_convert_control_type.rst

.. include :: struct_convert_time_type.rst

.. include :: struct_convert_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_convert_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/convert/C/convertt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/convert/C/convertt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/convert/C/converttf.c .

.. include :: ../../../src/convert/C/converttf.c
   :code: C
