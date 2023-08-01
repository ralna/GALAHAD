.. _doxid-index_sec:

GALAHAD SEC package
===================

.. module:: galahad.sec

.. include:: ../../Python/sec_intro.rst

.. toctree::
	:hidden:

	sec_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``sec_initialize_s``) 
are available with T as Float32.

.. include:: sec_functions.rst

available structures
--------------------

.. include :: struct_sec_control_type.rst

.. include :: struct_sec_inform_type.rst

|	:ref:`genindex`
