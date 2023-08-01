.. _doxid-index_roots:

GALAHAD ROOTS package
=====================

.. module:: galahad.roots

.. include:: ../../Python/roots_intro.rst

.. toctree::
	:hidden:

	roots_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``roots_initialize_s``) 
are available with T as Float32.

.. include:: roots_functions.rst

available structures
--------------------

.. include :: struct_roots_control_type.rst

.. include :: struct_roots_inform_type.rst

|	:ref:`genindex`
