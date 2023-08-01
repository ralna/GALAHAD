.. _doxid-index_ir:

GALAHAD IR package
==================

.. module:: galahad.ir

.. include:: ../../Python/ir_intro.rst

.. toctree::
	:hidden:

	ir_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``ir_initialize_s``) 
are available with T as Float32.

.. include:: ir_functions.rst

available structures
--------------------

.. include :: struct_ir_control_type.rst

.. include :: struct_ir_inform_type.rst

|	:ref:`genindex`
