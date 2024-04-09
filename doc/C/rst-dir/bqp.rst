.. _doxid-index_bqp:

GALAHAD BQP package
===================

.. module:: galahad.bqp

.. include:: ../../Python/bqp_intro.rst

.. include:: ../../Python/bqp_storage.rst

.. toctree::
	:hidden:

	bqp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bqp package must be 
called in the following order:

* :ref:`bqp_initialize <doxid-galahad__bqp_8h_1a4466621895dd2314f1b3c21b4bc7f615>` - provide default control parameters and set up initial data structures

* :ref:`bqp_read_specfile <doxid-galahad__bqp_8h_1a0e3ffdd29be95753292694c7619a43e6>` (optional) - override control values by reading replacement values from a file

* set up problem data structures and fixed values by caling one of
  
  * :ref:`bqp_import <doxid-galahad__bqp_8h_1a0cfa65e832fd80e3dfcf9e0c65a69e56>` - in the case that $H$ is explicitly available
  
  * :ref:`bqp_import_without_h <doxid-galahad__bqp_8h_1a9a99d880b3bfbcfb7b093756019c5f0e>` - in the case that only the effect of applying $H$ to a vector is possible

* :ref:`bqp_reset_control <doxid-galahad__bqp_8h_1a315ce83042f67a466cfdd868c27a2850>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`bqp_solve_given_h <doxid-galahad__bqp_8h_1acb5ad644890efe38b7cf7048d6297308>` - solve the problem using values of $H$
  
  * :ref:`bqp_solve_reverse_h_prod <doxid-galahad__bqp_8h_1a116b9b4ff28b9e2d18be0f0900ce2755>` - solve the problem by returning to the caller for products of $H$ with specified vectors

* :ref:`bqp_information <doxid-galahad__bqp_8h_1a75b662635f281148e9c19e12e0788362>` (optional) - recover information about the solution and solution process

* :ref:`bqp_terminate <doxid-galahad__bqp_8h_1a34db499197d1fd6fb78b294473796fbc>` - deallocate data structures

See the :ref:`examples <doxid-index_bqp_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: bqp_functions.rst

available structures
--------------------

.. include :: struct_bqp_control_type.rst

.. include :: struct_bqp_time_type.rst

.. include :: struct_bqp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_bqp_examples:

example calls
-------------

This is an example of how to use the package to solve a bound-constrained QP;
the code is available in $GALAHAD/src/bqp/C/bqpt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/bqp/C/bqpt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/bqp/C/bqptf.c .

.. include :: ../../../src/bqp/C/bqptf.c
   :code: C
