.. _doxid-index_sec:

GALAHAD SEC package
===================

.. module:: galahad.sec

.. include:: ../../Python/sec_intro.rst

.. include:: ../../Python/sec_storage.rst

.. toctree::
	:hidden:

	sec_functions.rst

callable functions
------------------

.. include:: sec_functions.rst

available structures
--------------------

.. include :: struct_sec_control_type.rst

.. include :: struct_sec_time_type.rst

.. include :: struct_sec_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_sec_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/sec/C/sect.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/sec/C/sect.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/sec/C/sectf.c .

.. include :: ../../../src/sec/C/sectf.c
   :code: C
