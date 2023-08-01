.. _doxid-index_convert:

GALAHAD CONVERT package
=======================

.. module:: galahad.convert

.. include:: ../../Python/convert_intro.rst

.. toctree::
	:hidden:

	convert_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``convert_initialize_s``) 
are available with T as Float32.

.. include:: convert_functions.rst

available structures
--------------------

.. include :: struct_convert_control_type.rst

.. include :: struct_convert_time_type.rst

.. include :: struct_convert_inform_type.rst

|	:ref:`genindex`

