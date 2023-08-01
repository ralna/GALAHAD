.. _doxid-index_scu:

GALAHAD SCU package
===================

.. module:: galahad.scu

.. include:: ../../Python/scu_intro.rst

.. toctree::
	:hidden:

	scu_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the scu package must be 
called in the following order:

* :ref:`scu_initialize <doxid->` - provide default control parameters and set up initial data structures

* :ref:`scu_read_specfile <doxid->` (optional) - override control values by reading replacement values from a file

* :ref:`scu_form_and_factorize <doxid->` - form and factorize the Schur-complement matrix :math:`S`

* :ref:`scu_solve_system <doxid->` - solve the block system (1)

* :ref:`scu_add_rows_and_cols <doxid->` (optional) - update the factors of the Schur-complement matrix when rows and columns are added to (1).

* :ref:`scu_delete_rows_and_cols <doxid->` (optional) - update the factors of the Schur-complement matrix when rows and columns are removed from (1).

* :ref:`scu_information <doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d>` (optional) - recover information about the solution and solution process

* :ref:`scu_terminate <doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad>` - deallocate data structures

See the :ref:`examples <doxid-index_scu_examples>` section for 
illustrations of use.

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``scu_initialize_s``) 
are available with T as Float32.

.. include:: scu_functions.rst

available structures
--------------------

.. include :: struct_scu_control_type.rst

.. include :: struct_scu_inform_type.rst

|	:ref:`genindex`
