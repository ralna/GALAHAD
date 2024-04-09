.. _doxid-index_uls:

GALAHAD ULS package
===================

.. module:: galahad.uls

.. include:: ../../Python/uls_intro.rst

.. include:: ../../Python/uls_storage.rst

.. toctree::
	:hidden:

	uls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the uls package must be 
called in the following order:

* :ref:`uls_initialize <doxid-galahad__uls_8h_1a7afb5f2dde112e60686a5527a8f37ca4>` - provide default control parameters and set up initial data structures

* :ref:`uls_read_specfile <doxid-galahad__uls_8h_1a5e2c9573bc8661114e9f073782b460ef>` (optional) - override control values by reading replacement values from a file

* :ref:`uls_factorize_matrix <doxid-galahad__uls_8h_1a6c0599479b84ee7d7c4ee7c473b76a83>` - set up matrix data structures, analyse the structure to choose a suitable order for factorization, and then factorize the matrix :math:`A`

* :ref:`uls_reset_control <doxid-galahad__uls_8h_1ad2ad6daa4d54d75e40fbe253f2bc5881>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`uls_solve_system <doxid-galahad__uls_8h_1a01d3e7c19415125c660eba51d99c7518>` - solve the linear system of equations :math:`Ax=b` or :math:`A^Tx=b`

* :ref:`uls_information <doxid-galahad__uls_8h_1ab41cc4ccba208d7de3a0ccbc4b4efbcf>` (optional) - recover information about the solution and solution process

* :ref:`uls_terminate <doxid-galahad__uls_8h_1a36b2ea1ade2cdd8bca238f46e9e98435>` - deallocate data structures

See the :ref:`examples <doxid-index_uls_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: uls_functions.rst

available structures
--------------------

.. include :: struct_uls_control_type.rst

.. include :: struct_uls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_uls_examples:

example calls
-------------

This is an example of how to use the package to solve a linear system;
the code is available in $GALAHAD/src/uls/C/ulst.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/uls/C/ulst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/uls/C/ulstf.c .

.. include :: ../../../src/uls/C/ulstf.c
   :code: C
