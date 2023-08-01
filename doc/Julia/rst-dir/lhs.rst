.. _doxid-index_lhs:

GALAHAD LHS package
===================

.. module:: galahad.lhs

.. include:: ../../Python/lhs_intro.rst

.. toctree::
	:hidden:

	lhs_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``lhs_initialize_s``) 
are available with T as Float32.

.. include:: lhs_functions.rst

available structures
--------------------

.. include :: struct_lhs_control_type.rst

.. include :: struct_lhs_inform_type.rst

|	:ref:`genindex`
