.. _doxid-index_sha:

GALAHAD SHA package
===================

.. module:: galahad.sha

.. include:: ../../Python/sha_intro.rst

.. include:: ../../Python/sha_storage.rst

.. toctree::
	:hidden:

	sha_functions.rst

callable functions
------------------

.. include:: sha_functions.rst

available structures
--------------------

.. include :: struct_sha_control_type.rst

.. include :: struct_sha_time_type.rst

.. include :: struct_sha_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_sha_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/sha/C/shat.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flaggeed by setting 
``control.f_indexing`` to ``false``. The floating-point type ``real_wp_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``GALAHAD_SINGLE`` is defined.

.. include :: ../../../src/sha/C/shat.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/sha/C/shatf.c .

.. include :: ../../../src/sha/C/shatf.c
   :code: C
