.. _doxid-index_lhs:

GALAHAD LHS package
===================

.. module:: galahad.lhs

.. include:: ../../Python/lhs_intro.rst

.. include:: ../../Python/lhs_storage.rst

.. toctree::
	:hidden:

	lhs_functions.rst

callable functions
------------------

.. include:: lhs_functions.rst

available structures
--------------------

.. include :: struct_lhs_control_type.rst

.. include :: struct_lhs_time_type.rst

.. include :: struct_lhs_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lhs_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/lhs/C/lhst.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/lhs/C/lhst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/lhs/C/lhstf.c .

.. include :: ../../../src/lhs/C/lhstf.c
   :code: C
