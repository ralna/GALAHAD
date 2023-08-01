.. _doxid-index_gls:

GALAHAD gls package
===================

.. module:: galahad.gls

.. include:: ../../Python/gls_intro.rst

.. toctree::
	:hidden:

	gls_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``gls_initialize_s``) 
are available with T as Float32.

.. include:: gls_functions.rst

available structures
--------------------

.. include :: struct_gls_control.rst

.. include :: struct_gls_ainfo.rst

.. include :: struct_gls_finfo.rst

.. include :: struct_gls_sinfo.rst

|	:ref:`genindex`

