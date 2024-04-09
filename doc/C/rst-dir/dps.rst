.. _doxid-index_dps:

GALAHAD DPS package
===================

.. module:: galahad.dps

.. include:: ../../Python/dps_intro.rst

.. include:: ../../Python/dps_storage.rst

.. toctree::
	:hidden:

	dps_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the dps package must be 
called in the following order:

* :ref:`dps_initialize <doxid-galahad__dps_8h_1a29104b1214a3af5b4dc76dca722250b4>` - provide default control parameters and set up initial data structures

* :ref:`dps_read_specfile <doxid-galahad__dps_8h_1a2b7fed0d89483ec1c49b517be04acdcf>` (optional) - override control values by reading replacement values from a file

* :ref:`dps_import <doxid-galahad__dps_8h_1a7bc05b1c7fd874e96481d0521262bdee>` - import control and matrix data structures

* :ref:`dps_reset_control <doxid-galahad__dps_8h_1a445d31a1c3e3aa63af85ceddd9769a5c>` (optional) - possibly change control parameters if a sequence of problems are being solved

* one of
  
  * :ref:`dps_solve_tr_problem <doxid-galahad__dps_8h_1a0ce2d73010a90e735fd98393d63cb1a5>` - solve the trust-region problem (1)
  
  * :ref:`dps_solve_rq_problem <doxid-galahad__dps_8h_1ae3baff5b8a4b59c37a6ada62dff67cc6>` - solve the regularized-quadratic problem (2)

* optionally one of
  
  * :ref:`dps_resolve_tr_problem <doxid-galahad__dps_8h_1af244a0e386040d5da2d11c3bd9d1e34d>` - resolve the trust-region problem (1) when the non-matrix data has changed
  
  * :ref:`dps_resolve_rq_problem <doxid-galahad__dps_8h_1a19e02a1d80eaedcb9e339f9963db352a>` - resolve the regularized-quadratic problem (2) when the non-matrix data has changed

* :ref:`dps_information <doxid-galahad__dps_8h_1a7617a692133347cb651f9a96244eb9f6>` (optional) - recover information about the solution and solution process

* :ref:`dps_terminate <doxid-galahad__dps_8h_1a1e67ac91c520fc4ec65df30e4140f57e>` - deallocate data structures

See the :ref:`examples <doxid-index_dps_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: dps_functions.rst

available structures
--------------------

.. include :: struct_dps_control_type.rst

.. include :: struct_dps_time_type.rst

.. include :: struct_dps_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_dps_examples:

example calls
-------------

This is an example of how to use the package to solve a trust-region subproblem;
the code is available in $GALAHAD/src/dps/C/dpst.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preproccesor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preproccesor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/dps/C/dpst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/dps/C/dpstf.c .

.. include :: ../../../src/dps/C/dpstf.c
   :code: C
