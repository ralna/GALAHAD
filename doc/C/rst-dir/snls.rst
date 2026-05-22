.. _doxid-index_snls:

GALAHAD SNLS package
====================

.. module:: galahad.snls

.. include:: ../../Python/snls_intro.rst

.. include:: ../../Python/snls_storage.rst

.. toctree::
	:hidden:

	snls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the snls package must be 
called in the following order:

To solve a given problem, functions from the snls package must be called in the following order:

* :ref:`snls_initialize <doxid-galahad__snls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f>` - provide default control parameters and set up initial data structures

* :ref:`snls_read_specfile <doxid-galahad__snls_8h_1adf9db7eff2fce137ae2abd2e013c47b3>` (optional) - override control values by reading replacement values from a file

* set up data structures by calling one of

  * :ref:`snls_import <doxid-galahad__snls_8h_1a3f0eb83fd31ee4108156f2e84176389d>` - set up problem data structures and fixed values when $J_r(x)$ is available

  * :ref:`snls_import_without_jac <doxid-galahad__snls_8h_1a3f0eb83fd31ee4108156f2e84176390d>` - set up problem data structures and fixed values when only products with $J_r(x)$ are available

* :ref:`snls_reset_control <doxid-galahad__snls_8h_1a07f0857c9923ad0f92d51ed00833afda>` (optional) - possibly change control parameters if a sequence of problems are being solved

* solve the problem by calling one of
  
  * :ref:`snls_solve_with_jac <doxid-galahad__snls_8h_1ae923c2e6afabb3563fe0998d45b715c4>` - solve using function calls to evaluate residual and Jacobian values
  
  * :ref:`snls_solve_with_jacprod <doxid-galahad__snls_8h_1a692ecbfaa428584e60aa4c33d7278a64>` - solve using function calls to evaluate residual  values and Jacobian-vector products
  
  * :ref:`snls_solve_reverse_with_jac <doxid-galahad__snls_8h_1a9ad89605640c53c33ddd5894b5e3edd1>` - solve returning to the calling program to obtain residual and Jacobian values, or
  
  * :ref:`snls_solve_reverse_with_jacprod <doxid-galahad__snls_8h_1a6dddd928c19adec0abf76bdb2d75da17>` - solve returning to the calling prorgram to obtain residual values and Jacobian-vector products

* :ref:`snls_information <doxid-galahad__snls_8h_1a765da96b0a1f3d07dab53cc3400c22d8>` (optional) - recover information about the solution and solution process

* :ref:`snls_terminate <doxid-galahad__snls_8h_1a7babe9112dfad1eb7b57b70135704ab0>` - deallocate data structures

See the :ref:`examples <doxid-index_snls_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: snls_functions.rst

available structures
--------------------

.. include :: struct_snls_control_type.rst

.. include :: struct_snls_time_type.rst

.. include :: struct_snls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_snls_examples:

example calls
-------------

This is an example of how to use the package to solve a simplex-constrained 
nonlinear least-squares problem; the code is available in $GALAHAD/src/snls/C/snlst.c .

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/snls/C/snlst.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/snls/C/snlstf.c .

.. include :: ../../../src/snls/C/snlstf.c
   :code: C
