.. _doxid-index_psls:

GALAHAD PSLS package
====================

.. module:: galahad.psls

.. include:: ../../Python/psls_intro.rst

.. include:: s_storage.rst

.. toctree::
	:hidden:

	psls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the psls package must be
called in the following order:

* :ref:`psls_initialize <doxid-galahad__psls_8h_1af5cb66dbf5b9e4f094e2e0a29631fd1b>` - provide default control parameters and set up initial data structures

* :ref:`psls_read_specfile <doxid-galahad__psls_8h_1a34b978446b6aa5636f9e6efc18860366>` (optional) - override control values by reading replacement values from a file

* :ref:`psls_import <doxid-galahad__psls_8h_1a3ff902c85fb82f1929a93514bb63c5d6>` - set up matrix data structures for $A$ prior to solution

* :ref:`psls_reset_control <doxid-galahad__psls_8h_1a90493b62c689237c97fe4aea665cd0ab>` (optional) - possibly change control parameters if a sequence of problems are being solved

* one of

  * :ref:`psls_form_preconditioner <doxid-galahad__psls_8h_1a9cd4c449dcc5133932972866fd58cfc1>` - form and factorize a preconditioner $P$ of the matrix $A$

  * :ref:`psls_form_subset_preconditioner <doxid-galahad__psls_8h_1a75fa79fcbe08ab367b9fa0b7f39adf65>` - form and factorize a preconditioner $P$ of a symmetric submatrix of the matrix $A$

* :ref:`psls_update_preconditioner <doxid-galahad__psls_8h_1a42a8097e64b527cff18ab66c07a32d1d>` (optional) - update the preconditioner $P$ when rows (amd columns) are removed

* :ref:`psls_apply_preconditioner <doxid-galahad__psls_8h_1a1bae97d4a0e63bce7380422ed83306e8>` - solve the linear system of equations $Px=b$

* :ref:`psls_information <doxid-galahad__psls_8h_1ace5f302a9ccb0c3f8c29b28b42da7793>` (optional) - recover information about the preconditioner and solution process

* :ref:`psls_terminate <doxid-galahad__psls_8h_1ab62a2e262e7466fac3a2dc8cd300720d>` - deallocate data structures

See the :ref:`examples <doxid-index_psls_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: psls_functions.rst

available structures
--------------------

.. include :: struct_psls_control_type.rst

.. include :: struct_psls_time_type.rst

.. include :: struct_psls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_psls_examples:

example calls
-------------

This is an example of how to use the package to solve a definite linear system;
the code is available in $GALAHAD/src/psls/Julia/test_psls.jl .
A variety of supported matrix storage formats are shown.

.. include :: ../../../src/psls/Julia/test_psls.jl
   :code: julia
