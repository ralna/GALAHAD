.. _doxid-index_lms:

GALAHAD LMS package
===================

.. module:: galahad.lms

.. include:: ../../Python/lms_intro.rst

.. toctree::
	:hidden:

	lms_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``lms_initialize_s``) 
are available with T as Float32.

.. include:: lms_functions.rst

available structures
--------------------

.. include :: struct_lms_control_type.rst

.. include :: struct_lms_time_type.rst

.. include :: struct_lms_inform_type.rst

|	:ref:`genindex`

