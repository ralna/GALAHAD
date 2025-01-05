.. _doxid-index_sls:

GALAHAD SLS package
===================

.. module:: galahad.sls

.. include:: ../../Python/sls_intro.rst

.. include:: s_storage.rst

.. toctree::
	:hidden:

	sls_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the sls package must be
called in the following order:

* :ref:`sls_initialize <doxid-galahad__sls_8h_1a1d8a0c73587ca6d7f5333d41b3e2472a>` - provide default control parameters and set up initial data structures

* :ref:`sls_read_specfile <doxid-galahad__sls_8h_1ada1e7b9ed799335702f85a551b64bf88>` (optional) - override control values by reading replacement values from a file

* :ref:`sls_analyse_matrix <doxid-galahad__sls_8h_1a380a7f50cc71c705d15a791acde946cf>` - set up matrix data structures and analyse the structure to choose a suitable order for factorization

* :ref:`sls_reset_control <doxid-galahad__sls_8h_1aacc344b8cdf0b1c27965f191382372e4>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`sls_factorize_matrix <doxid-galahad__sls_8h_1ab6666f5eb7b0bdbbc9c9b52b7a2e2c41>` - form and factorize the matrix $A$

* one of

  * :ref:`sls_solve_system <doxid-galahad__sls_8h_1a1b3e7546b59b06160c51e16b6781bc0b>` - solve the linear system of equations $Ax=b$

  * :ref:`sls_partial_solve_system <doxid-galahad__sls_8h_1ac66dc50d8b54acab90d70ae649b92905>` - solve a linear system $Mx=b$ involving one of the matrix factors $M$ of $A$

* :ref:`sls_information <doxid-galahad__sls_8h_1a0ca4a126813c3aafac9d791a152b233c>` (optional) - recover information about the solution and solution process

* :ref:`sls_terminate <doxid-galahad__sls_8h_1aa5aafa378e3500ce31783e13c3395d30>` - deallocate data structures

See the :ref:`examples <doxid-index_sls_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: sls_functions.rst

available structures
--------------------

.. include :: struct_sls_control_type.rst

.. include :: struct_sls_time_type.rst

.. include :: struct_sls_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_sls_examples:

example calls
-------------

This is an example of how to use the package to solve a symmetric system of 
linear equations; the code is available in $GALAHAD/src/sls/Julia/test_sls.jl .
A variety of supported  matrix storage formats are shown.

.. include :: ../../../src/sls/Julia/test_sls.jl
   :code: julia
