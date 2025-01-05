.. index:: pair: struct; dqp_inform_type
.. _doxid-structdqp__inform__type:

dqp_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct dqp_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          cg_iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          threads::INT
          obj::T
          primal_infeasibility::T
          dual_infeasibility::T
          complementary_slackness::T
          non_negligible_pivot::T
          feasible::Bool
          checkpointsIter::NTuple{16,INT}
          checkpointsTime::NTuple{16,T}
          time::dqp_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sls_inform::sls_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}
          gltr_inform::gltr_inform_type{T,INT}
          scu_status::INT
          scu_inform::scu_inform_type{INT}
          rpd_inform::rpd_inform_type{INT}
	
.. _details-structdqp__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See DQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structdqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structdqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structdqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structdqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structdqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structdqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structdqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; threads
.. _doxid-structdqp__inform__type_1a4f987a98d3e1221916748962e45399fe:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structdqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by DQP_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structdqp__inform__type_1a2bce6cd733ae08834689fa66747f53b9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structdqp__inform__type_1a979cebdf2e5f1e043f48a615a46b0299:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structdqp__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structdqp__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot that was not judged to be zero when detecting linearly dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structdqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structdqp__inform__type_1acb0789a29239327ab8a4e929e0fbc65b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structdqp__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structdqp__inform__type_1ab2d91973c7fda1aa150ef72bb22842f6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dqp_time_type<doxid-structdqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structdqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structdqp__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structdqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structdqp__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; scu_status
.. _doxid-structdqp__inform__type_1a25bf1e7f86c2b4f4836aa4de40019815:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT scu_status

inform parameters for SCU

.. index:: pair: variable; scu_inform
.. _doxid-structdqp__inform__type_1a0b702af94f05b9d4bb2bb6416f2498ee:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; rpd_inform
.. _doxid-structdqp__inform__type_1a823701505feea7615e9f8995769d8b60:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

