.. _doxid-index_wcp:

GALAHAD WCP package
===================

.. module:: galahad.wcp

.. include:: ../../Python/wcp_intro.rst

.. include:: a_storage.rst

.. toctree::
	:hidden:

	wcp_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the wcp package must be
called in the following order:

* :ref:`wcp_initialize <doxid-galahad__wcp_8h_1a78c76e090a879684ca1fa3ab17f55f34>` - provide default control parameters and set up initial data structures

* :ref:`wcp_read_specfile <doxid-galahad__wcp_8h_1af50523066dbb40bc7f955e0ef35881a9>` (optional) - override control values by reading replacement values from a file

* :ref:`wcp_import <doxid-galahad__wcp_8h_1a91b5d7b341c1333669564a1abacc2ad9>` - set up problem data structures and fixed values

* :ref:`wcp_reset_control <doxid-galahad__wcp_8h_1a4b6ac93a46f87e3e986286c415155dd3>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`wcp_find_wcp <doxid-galahad__wcp_8h_1a5ca84b359a491ced6fdb1c0927b25243>` - find a well-centered point

* :ref:`wcp_information <doxid-galahad__wcp_8h_1aa3f76e788325ffff83f98dffa7ab8eb2>` (optional) - recover information about the solution and solution process

* :ref:`wcp_terminate <doxid-galahad__wcp_8h_1a0b1cc55b8418826d80e4435ab555e256>` - deallocate data structures

See the :ref:`examples <doxid-index_wcp_examples>` section for
illustrations of use.

.. include:: irt.rst

.. include:: wcp_functions.rst


available structures
--------------------

.. include :: struct_wcp_control_type.rst

.. include :: struct_wcp_time_type.rst

.. include :: struct_wcp_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_wcp_examples:

example calls
-------------

This is an example of how to use the package to find a well-centred point;
the code is available in $GALAHAD/src/wcp/Julia/test_wcp.jl .
A variety of supported Hessian and constraint matrix storage formats are shown.

.. include :: ../../../src/wcp/Julia/test_wcp.jl
   :code: julia
