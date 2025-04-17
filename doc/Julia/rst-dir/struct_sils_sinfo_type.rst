.. index:: pair: struct; sils_sinfo_type
.. _doxid-structsils__sinfo__type:

sils_sinfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sils_sinfo_type{T,INT}
          flag::INT
          stat::INT
          cond::T
          cond2::T
          berr::T
          berr2::T
          error::T

.. _details-structsils__sinfo__type:

detailed documentation
----------------------

sinfo derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__sinfo__type_flag:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; stat
.. _doxid-structsils__sinfo__type_stat:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

STAT value after allocate failure.

.. index:: pair: variable; cond
.. _doxid-structsils__sinfo__type_cond:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cond

Condition number of matrix (category 1 eqs)

.. index:: pair: variable; cond2
.. _doxid-structsils__sinfo__type_cond2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cond2

Condition number of matrix (category 2 eqs)

.. index:: pair: variable; berr
.. _doxid-structsils__sinfo__type_berr:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T berr

Backward error for the system (category 1 eqs)

.. index:: pair: variable; berr2
.. _doxid-structsils__sinfo__type_berr2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T berr2

Backward error for the system (category 2 eqs)

.. index:: pair: variable; error
.. _doxid-structsils__sinfo__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T error

Estimate of forward error.

