.. index:: pair: struct; lpb_inform_type
.. _doxid-structlpb__inform__type:

lpb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lpb_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          nbacts::INT
          threads::INT
          obj::T
          primal_infeasibility::T
          dual_infeasibility::T
          complementary_slackness::T
          init_primal_infeasibility::T
          init_dual_infeasibility::T
          init_complementary_slackness::T
          potential::T
          non_negligible_pivot::T
          feasible::Bool
          checkpointsIter::NTuple{16,INT}
          checkpointsTime::NTuple{16,T}
          time::lpb_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}
          fit_inform::fit_inform_type{INT}
          roots_inform::roots_inform_type{INT}
          cro_inform::cro_inform_type{T,INT}
          rpd_inform::rpd_inform_type{INT}
	
.. _details-structlpb__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlpb__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See LPB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlpb__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlpb__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlpb__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structlpb__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structlpb__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structlpb__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structlpb__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structlpb__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; threads
.. _doxid-structlpb__inform__type_1a4f987a98d3e1221916748962e45399fe:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structlpb__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by LPB_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structlpb__inform__type_1a2bce6cd733ae08834689fa66747f53b9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structlpb__inform__type_1a979cebdf2e5f1e043f48a615a46b0299:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structlpb__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; init_primal_infeasibility
.. _doxid-structlpb__inform__type_1a85355c90bdd4cdd2e09bfed7fd9f66e1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_primal_infeasibility

these values at the initial point (needed bg GALAHAD_CLPB)

.. index:: pair: variable; init_dual_infeasibility
.. _doxid-structlpb__inform__type_1a8a03c79f840170d644609f1fb95d06e3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_dual_infeasibility

see init_primal_infeasibility

.. index:: pair: variable; init_complementary_slackness
.. _doxid-structlpb__inform__type_1adbb8b72850a5e57700268fc582064615:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_complementary_slackness

see init_primal_infeasibility

.. index:: pair: variable; potential
.. _doxid-structlpb__inform__type_1a85f37aa42c9e051ea61ae035ff63059e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structlpb__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linearly dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structlpb__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structlpb__inform__type_1acb0789a29239327ab8a4e929e0fbc65b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structlpb__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structlpb__inform__type_1a3f1797234f099e3102a92630951c9a4e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lpb_time_type<doxid-structlpb__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structlpb__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structlpb__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; fit_inform
.. _doxid-structlpb__inform__type_1ac6efa45e989564727014956bf3e00deb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

.. index:: pair: variable; roots_inform
.. _doxid-structlpb__inform__type_1a68574d04a336f7be88a151fa8b975885:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

return information from ROOTS

.. index:: pair: variable; cro_inform
.. _doxid-structlpb__inform__type_1a594c21272a0c3d75d5ffa712f1d8f971:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`cro_inform_type<doxid-structcro__inform__type>` cro_inform

inform parameters for CRO

.. index:: pair: variable; rpd_inform
.. _doxid-structlpb__inform__type_1a823701505feea7615e9f8995769d8b60:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

