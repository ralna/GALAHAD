
.. _doxid-index_sha:

GALAHAD SHA package
===================

.. module:: galahad.sha

.. include:: ../../Python/sha_intro.rst

.. toctree::
	:hidden:

	sha_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants
(with the additional suffix ``_s``, e.g., ``sha_initialize_s``)
are available with T as Float32.

.. include:: sha_functions.rst

available structures
--------------------

.. include :: struct_sha_control_type.rst

.. include :: struct_sha_inform_type.rst

|	:ref:`genindex`
