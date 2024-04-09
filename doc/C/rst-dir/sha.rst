.. _doxid-index_sha:

GALAHAD SHA package
===================

.. module:: galahad.sha

.. include:: ../../Python/sha_intro.rst

.. toctree::
	:hidden:

	sha_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the sha package must be called in the following order:

* :ref:`sha_initialize <doxid-galahad__sha_8h_initialize>` - provide default control parameters and set up initial data structures

* :ref:`sha_read_specfile <doxid-galahad__sha_8h_read_specfile>` (optional) - override control values by reading replacement values from a file

* :ref:`sha_analyse_matrix <doxid-galahad__sha_8h_analyse_matrix>` - set up problem data structures and generate information that will be used when estimating Hessian values

* :ref:`sha_recover_matrix <doxid-galahad__sha_8h_recover_matrix>` - recover the Hessian approximation
  
* :ref:`sha_information <doxid-galahad__sha_8h_information>` (optional) - recover information about the solution and solution process

* :ref:`sha_terminate <doxid-galahad__sha_8h_terminate>` - deallocate data structures

See the :ref:`examples <doxid-index_sha_examples>` section for illustrations 
of use.

callable functions
------------------

.. include:: sha_functions.rst

available structures
--------------------

.. include :: struct_sha_control_type.rst

.. include :: struct_sha_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_sha_examples:

example calls
-------------

This is an example of how to use the package to find a Hessian approximation
using gradient differences; the code is available in $GALAHAD/src/sha/C/shat.c .

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/sha/C/shat.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/sha/C/shatf.c .

.. include :: ../../../src/sha/C/shatf.c
   :code: C
