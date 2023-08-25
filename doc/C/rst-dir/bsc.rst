.. _doxid-index_bsc:

GALAHAD BSC package
===================

.. module:: galahad.bsc

.. include:: ../../Python/bsc_intro.rst

.. toctree::
	:hidden:

	bsc_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the bsc package must be 
called in the following order:

* :ref:`bsc_initialize <doxid-galahad__bsc_8h_1a32dd948f5ce268b0cdb340f435819c8e>` - provide default control parameters and set up initial data structures

* :ref:`bsc_read_specfile <doxid->` (optional) - override control values by reading replacement values from a file

* :ref:`bsc_import <doxid->` - set up matrix data structures for $A$.

* :ref:`bsc_reset_control <doxid->` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`bsc_form <doxid->` - form the Schur complement $S$

* :ref:`bsc_information <doxid-galahad__bsc_8h_1a4c0ae2150d39c240539e1d3be836b0af>` (optional) - recover information about the process

* :ref:`bsc_terminate <doxid-galahad__bsc_8h_1a3a8a2f875e681225b4851d060e310271>` - deallocate data structures

See the :ref:`examples <doxid-index_bsc_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: bsc_functions.rst

available structures
--------------------

.. include :: struct_bsc_control_type.rst

.. include :: struct_bsc_inform_type.rst

|	:ref:`genindex`

