.. _doxid-index_roots:

GALAHAD ROOTS package
=====================

.. module:: galahad.roots

.. include:: ../../Python/roots_intro.rst

.. include:: ../../Python/roots_storage.rst

.. toctree::
	:hidden:

	roots_functions.rst

callable functions
------------------

.. include:: roots_functions.rst

available structures
--------------------

.. include :: struct_roots_control_type.rst

.. include :: struct_roots_time_type.rst

.. include :: struct_roots_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_roots_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/roots/C/rootst.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/roots/C/rootst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/roots/C/rootstf.c .

.. include :: ../../../src/roots/C/rootstf.c
   :code: C
