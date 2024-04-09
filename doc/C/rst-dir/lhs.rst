.. _doxid-index_lhs:

GALAHAD LHS package
===================

.. module:: galahad.lhs

.. include:: ../../Python/lhs_intro.rst

.. toctree::
	:hidden:

	lhs_functions.rst

callable functions
------------------

.. include:: lhs_functions.rst

available structures
--------------------

.. include :: struct_lhs_control_type.rst

.. include :: struct_lhs_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_lhs_examples:

example calls
-------------

This is an example of how to use the package to ... ;
the code is available in $GALAHAD/src/lhs/C/lhst.c .

The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/lhs/C/lhst.c
   :code: C
