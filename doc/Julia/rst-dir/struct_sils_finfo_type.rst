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
.. _doxid-structsils__finfo__type_1adf916204820072417ed73a32de1cefcf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__finfo__type_1a4628f2fb17af64608416810cc4e5a9d0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more

More information on failure.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__finfo__type_1a1a03f6fb5030d0b1d6bf3703d8c24cc4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxfrt

Largest front size.

.. index:: pair: variable; nebdu
.. _doxid-structsils__finfo__type_1ad711b100f86e360e8a840d058d96671a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nebdu

Number of entries in factors.

.. index:: pair: variable; nrlbdu
.. _doxid-structsils__finfo__type_1af2c296d8547ed8e4278a63772f775e50:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrlbdu

Number of reals that hold factors.

.. index:: pair: variable; nirbdu
.. _doxid-structsils__finfo__type_1a8fbe1e93500e9552d69c42a20dbeb7ba:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirbdu

Number of integers that hold factors.

.. index:: pair: variable; nrltot
.. _doxid-structsils__finfo__type_1a87d1fb870bb3dc94213c3adf550a7a3a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__finfo__type_1af19131d647173a929ed0594f33c46999:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__finfo__type_1a812f6ee1b4bd452745b3876b084cd4af:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__finfo__type_1a2db255a81b52a84b3fec32d8c1a4ec17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nirnec

Size for iw with compression.

.. index:: pair: variable; ncmpbr
.. _doxid-structsils__finfo__type_1a1f120d69b5f9dad9acb22de7946169ac:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ncmpbr

Number of compresses of real data.

.. index:: pair: variable; ncmpbi
.. _doxid-structsils__finfo__type_1a56e6baa407d862416316a7ef3edc345c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ncmpbi

Number of compresses of integer data.

.. index:: pair: variable; ntwo
.. _doxid-structsils__finfo__type_1aa8a36bb6a93ac6d099aef27c76ccc1f8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ntwo

Number of 2x2 pivots.

.. index:: pair: variable; neig
.. _doxid-structsils__finfo__type_1a5a9c67ca97b64a81e04f18d037242075:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neig

Number of negative eigenvalues.

.. index:: pair: variable; delay
.. _doxid-structsils__finfo__type_1a6f1be1f780ff54ec75b41451cd4d90bd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT delay

Number of delayed pivots (total)

.. index:: pair: variable; signc
.. _doxid-structsils__finfo__type_1a3321ebac07d2b23b0fcc04ac6f0129bf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT signc

Number of pivot sign changes when control.pivoting=3.

.. index:: pair: variable; nstatic
.. _doxid-structsils__finfo__type_1a4f8d909bc8fbafc899cd820077eb802b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nstatic

Number of static pivots chosen.

.. index:: pair: variable; modstep
.. _doxid-structsils__finfo__type_1a7fdac7d8f75bd2eff8fac17e9a36da64:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT modstep

First pivot modification when control.pivoting=4.

.. index:: pair: variable; rank
.. _doxid-structsils__finfo__type_1a6cfd95afd0afebd625b889fb6e58371c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

Rank of original factorization.

.. index:: pair: variable; stat
.. _doxid-structsils__finfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__finfo__type_1a6dfcf1d95eca288e459a114d6418d6f1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT faulty

legacy component, now not used

.. index:: pair: variable; step
.. _doxid-structsils__finfo__type_1abc16e65f240ed0c8f3e876e8732c0a33:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT step

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__finfo__type_1a5514c74cf880eb762c480fe9c94b45dd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opsa



.. _doxid-galahad__sils_8h_1autotoc_md0:

operations in assembly
~~~~~~~~~~~~~~~~~~~~~~

.. index:: pair: variable; opse
.. _doxid-structsils__finfo__type_1ae4d20d784bebe9b7f04a435f016e20d6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opse

number of operations in elimination

.. index:: pair: variable; opsb
.. _doxid-structsils__finfo__type_1a13983da267912e6dd3d3aa630542bba7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T opsb

Additional number of operations for BLAS.

.. index:: pair: variable; maxchange
.. _doxid-structsils__finfo__type_1a1082a3beb83c2223482de252ca6f12cb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maxchange

Largest control.pivoting=4 modification.

.. index:: pair: variable; smin
.. _doxid-structsils__finfo__type_1a36400de40b86ef41ccf65900ae74e219:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T smin

Minimum scaling factor.

.. index:: pair: variable; smax
.. _doxid-structsils__finfo__type_1a47957579e4f203ac77f60ea3eace2467:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T smax

Maximum scaling factor.

