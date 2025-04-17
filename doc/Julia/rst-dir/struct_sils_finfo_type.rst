.. index:: pair: struct; sils_finfo_type
.. _doxid-structsils__finfo__type:

sils_finfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sils_finfo_type{T,INT}
          flag::INT
          more::INT
          maxfrt::INT
          nebdu::INT
          nrlbdu::INT
          nirbdu::INT
          nrltot::INT
          nirtot::INT
          nrlnec::INT
          nirnec::INT
          ncmpbr::INT
          ncmpbi::INT
          ntwo::INT
          neig::INT
          delay::INT
          signc::INT
          nstatic::INT
          modstep::INT
          rank::INT
          stat::INT
          faulty::INT
          step::INT
          opsa::T
          opse::T
          opsb::T
          maxchange::T
          smin::T
          smax::T

.. _details-structsils__finfo__type:

detailed documentation
----------------------

finfo derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__finfo__type_flag:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__finfo__type_more:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more

More information on failure.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__finfo__type_maxfrt:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxfrt

Largest front size.

.. index:: pair: variable; nebdu
.. _doxid-structsils__finfo__type_nebdu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nebdu

Number of entries in factors.

.. index:: pair: variable; nrlbdu
.. _doxid-structsils__finfo__type_nrlbdu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrlbdu

Number of reals that hold factors.

.. index:: pair: variable; nirbdu
.. _doxid-structsils__finfo__type_nirbdu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirbdu

Number of integers that hold factors.

.. index:: pair: variable; nrltot
.. _doxid-structsils__finfo__type_nrltot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__finfo__type_nirtot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__finfo__type_nrlnec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__finfo__type_nirnec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirnec

Size for iw with compression.

.. index:: pair: variable; ncmpbr
.. _doxid-structsils__finfo__type_ncmpbr:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ncmpbr

Number of compresses of real data.

.. index:: pair: variable; ncmpbi
.. _doxid-structsils__finfo__type_ncmpbi:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ncmpbi

Number of compresses of integer data.

.. index:: pair: variable; ntwo
.. _doxid-structsils__finfo__type_ntwo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ntwo

Number of 2x2 pivots.

.. index:: pair: variable; neig
.. _doxid-structsils__finfo__type_neig:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neig

Number of negative eigenvalues.

.. index:: pair: variable; delay
.. _doxid-structsils__finfo__type_delay:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT delay

Number of delayed pivots (total)

.. index:: pair: variable; signc
.. _doxid-structsils__finfo__type_signc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT signc

Number of pivot sign changes when control.pivoting=3.

.. index:: pair: variable; nstatic
.. _doxid-structsils__finfo__type_nstatic:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nstatic

Number of static pivots chosen.

.. index:: pair: variable; modstep
.. _doxid-structsils__finfo__type_modstep:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT modstep

First pivot modification when control.pivoting=4.

.. index:: pair: variable; rank
.. _doxid-structsils__finfo__type_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

Rank of original factorization.

.. index:: pair: variable; stat
.. _doxid-structsils__finfo__type_stat:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__finfo__type_faulty:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT faulty

legacy component, now not used

.. index:: pair: variable; step
.. _doxid-structsils__finfo__type_step:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT step

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__finfo__type_opsa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opsa



.. _doxid-galahad__sils_8h_1autotoc_md0:

operations in assembly
~~~~~~~~~~~~~~~~~~~~~~

.. index:: pair: variable; opse
.. _doxid-structsils__finfo__type_opse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opse

number of operations in elimination

.. index:: pair: variable; opsb
.. _doxid-structsils__finfo__type_opsb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opsb

Additional number of operations for BLAS.

.. index:: pair: variable; maxchange
.. _doxid-structsils__finfo__type_maxchange:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maxchange

Largest control.pivoting=4 modification.

.. index:: pair: variable; smin
.. _doxid-structsils__finfo__type_smin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T smin

Minimum scaling factor.

.. index:: pair: variable; smax
.. _doxid-structsils__finfo__type_smax:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T smax

Maximum scaling factor.

