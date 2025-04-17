.. index:: pair: struct; sils_ainfo_type
.. _doxid-structsils__ainfo__type:

sils_ainfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sils_ainfo_type{T,INT}
          flag::INT
          more::INT
          nsteps::INT
          nrltot::INT
          nirtot::INT
          nrlnec::INT
          nirnec::INT
          nrladu::INT
          niradu::INT
          ncmpa::INT
          oor::INT
          dup::INT
          maxfrt::INT
          stat::INT
          faulty::INT
          opsa::T
          opse::T

.. _details-structsils__ainfo__type:

detailed documentation
----------------------

ainfo derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__ainfo__type_flag:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__ainfo__type_more:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more

More information on failure.

.. index:: pair: variable; nsteps
.. _doxid-structsils__ainfo__type_nsteps:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nsteps

Number of elimination steps.

.. index:: pair: variable; nrltot
.. _doxid-structsils__ainfo__type_nrltot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__ainfo__type_nirtot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__ainfo__type_nrlnec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__ainfo__type_nirnec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirnec

Size for iw with compression.

.. index:: pair: variable; nrladu
.. _doxid-structsils__ainfo__type_nrladu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrladu

Number of reals to hold factors.

.. index:: pair: variable; niradu
.. _doxid-structsils__ainfo__type_niradu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT niradu

Number of integers to hold factors.

.. index:: pair: variable; ncmpa
.. _doxid-structsils__ainfo__type_ncmpa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ncmpa

Number of compresses.

.. index:: pair: variable; oor
.. _doxid-structsils__ainfo__type_oor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT oor

Number of indices out-of-range.

.. index:: pair: variable; dup
.. _doxid-structsils__ainfo__type_dup:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dup

Number of duplicates.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__ainfo__type_maxfrt:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxfrt

Forecast maximum front size.

.. index:: pair: variable; stat
.. _doxid-structsils__ainfo__type_stat:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__ainfo__type_faulty:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT faulty

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__ainfo__type_opsa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opsa

Anticipated number of operations in assembly.

.. index:: pair: variable; opse
.. _doxid-structsils__ainfo__type_opse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opse

Anticipated number of operations in elimination.

