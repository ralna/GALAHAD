.. _doxid-index_ir:

GALAHAD IR package
==================

.. module:: galahad.ir

.. include:: ../../Python/ir_intro.rst

.. include:: ../../Python/ir_storage.rst

.. toctree::
	:hidden:

	ir_functions.rst

callable functions
------------------

.. include:: ir_functions.rst

available structures
--------------------

.. include :: struct_ir_control_type.rst

.. include :: struct_ir_time_type.rst

.. include :: struct_ir_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_ir_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/ir/C/irt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/ir/C/irt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/ir/C/irtf.c .

.. include :: ../../../src/ir/C/irtf.c
   :code: C
