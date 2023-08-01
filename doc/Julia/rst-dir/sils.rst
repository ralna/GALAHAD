.. _doxid-index_sils:

GALAHAD SILS package
====================

.. module:: galahad.sils

.. include:: ../../Python/sils_intro.rst

.. toctree::
	:hidden:

	sils_functions.rst

parametric real type T
----------------------

Below, the symbol T refers to a parametric real type that may be Float32
(single precision) or Float64 (double precision).
Calable functions as described are with T as Float64, but variants 
(with the additional suffix ``_s``, e.g., ``sils_initialize_s``) 
are available with T as Float32.

.. include:: sils_functions.rst

available structures
--------------------

.. include :: struct_sils_control_type.rst

.. include :: struct_sils_ainfo_type.rst

.. include :: struct_sils_finfo_type.rst

.. include :: struct_sils_sinfo_type.rst

|	:ref:`genindex`

