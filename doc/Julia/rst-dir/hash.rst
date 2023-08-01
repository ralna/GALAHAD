.. _doxid-index_hash:

GALAHAD HASH package
====================

.. module:: galahad.hash

.. include:: ../../Python/hash_intro.rst

.. toctree::
	:hidden:

	hash_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``hash_initialize_s``) 
are available with T as Float32.

.. include:: hash_functions.rst

available structures
--------------------

.. include :: struct_hash_control_type.rst

.. include :: struct_hash_inform_type.rst

|	:ref:`genindex`
