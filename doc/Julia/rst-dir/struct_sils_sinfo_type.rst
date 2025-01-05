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
.. _doxid-structsils__sinfo__type_1adf916204820072417ed73a32de1cefcf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; stat
.. _doxid-structsils__sinfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

STAT value after allocate failure.

.. index:: pair: variable; cond
.. _doxid-structsils__sinfo__type_1a006d728493fbea61aabf1e6229e34185:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cond

Condition number of matrix (category 1 eqs)

.. index:: pair: variable; cond2
.. _doxid-structsils__sinfo__type_1ae6b598341b9634df4e446be3de0ed839:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cond2

Condition number of matrix (category 2 eqs)

.. index:: pair: variable; berr
.. _doxid-structsils__sinfo__type_1ad2150d4466031c9e63281a146e5ccd03:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T berr

Backward error for the system (category 1 eqs)

.. index:: pair: variable; berr2
.. _doxid-structsils__sinfo__type_1ade02e126e145400e9ead3c3f3bc06dab:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T berr2

Backward error for the system (category 2 eqs)

.. index:: pair: variable; error
.. _doxid-structsils__sinfo__type_1a2b7e3bae2c2111a08302ba1dc7f14cef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T error

Estimate of forward error.

