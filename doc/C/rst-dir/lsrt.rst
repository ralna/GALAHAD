.. _doxid-index_lsrt:

GALAHAD LSRT package
====================

.. module:: galahad.lsrt

.. include:: ../../Python/lsrt_intro.rst

.. toctree::
	:hidden:

	lsrt_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the lsrt package must be 
called in the following order:

* :ref:`lsrt_initialize <doxid-galahad__lsrt_8h_1a9c5c14ddb34a5ea1becd133837da6544>` - provide default control parameters and set up initial data structures

* :ref:`lsrt_read_specfile <doxid-galahad__lsrt_8h_1a07c4c60e1ab6ae67a4da710e2ed01ff0>` (optional) - override control values by reading replacement values from a file

* :ref:`lsrt_import_control <doxid-galahad__lsrt_8h_1a09e39db33990f0c8a66480f54ba80f09>` - import control parameters prior to solution

* :ref:`lsrt_solve_problem <doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992>` - solve the problem by reverse communication, a sequence of calls are made under control of a status parameter, each exit either asks the user to provide additional informaton and to re-enter, or reports that either the solution has been found or that an error has occurred

* :ref:`lsrt_information <doxid-galahad__lsrt_8h_1ad3895aabdb7f18f84d209b02287872be>` (optional) - recover information about the solution and solution process

* :ref:`lsrt_terminate <doxid-galahad__lsrt_8h_1ac3a3d73e2686538802563c795a1afff4>` - deallocate data structures

See the :ref:`examples <doxid-index_lsrt_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: lsrt_functions.rst

available structures
--------------------

.. include :: struct_lsrt_control_type.rst

.. include :: struct_lsrt_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lsrt_examples:

example calls
-------------

This is an example of how to use the package to solve a regularized linear 
least-squares problem; the code is available in $GALAHAD/src/lsrt/C/lsrtt.c .

The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/lsrt/C/lsrtt.c
   :code: C
