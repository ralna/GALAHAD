.. _doxid-index_hash:

GALAHAD HASH package
====================

.. module:: galahad.hash

.. include:: ../../Python/hash_intro.rst

.. include:: ../../Python/hash_storage.rst

.. toctree::
	:hidden:

	hash_functions.rst

callable functions
------------------

.. include:: hash_functions.rst

available structures
--------------------

.. include :: struct_hash_control_type.rst

.. include :: struct_hash_time_type.rst

.. include :: struct_hash_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_hash_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/hash/C/hasht.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/hash/C/hasht.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/hash/C/hashtf.c .

.. include :: ../../../src/hash/C/hashtf.c
   :code: C
