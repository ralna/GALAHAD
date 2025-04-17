.. index:: pair: struct; clls_inform_type
.. _doxid-structclls__inform__type:

clls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct clls_inform_type{T,INT}
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
          non_negligible_pivot::T
          feasible::Bool
          checkpointsIter::NTuple{16,INT}
          checkpointsTime::NTuple{16,T}
          time::clls_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sls_inform::sls_inform_type{T,INT}
          sls_pounce_inform::sls_inform_type{T,INT}
          fit_inform::fit_inform_type{INT}
          roots_inform::roots_inform_type
          cro_inform::cro_inform_type{T,INT}
          rpd_inform::rpd_inform_type{INT}

.. _details-structclls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structclls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See CLLS_solve for details.

.. index:: pair: variable; alloc_status
.. _doxid-structclls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structclls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; iter
.. _doxid-structclls__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structclls__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structclls__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structclls__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structclls__inform__type_nfacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structclls__inform__type_nbacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; threads
.. _doxid-structclls__inform__type_threads:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structclls__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by CLLS_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structclls__inform__type_primal_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structclls__inform__type_dual_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structclls__inform__type_complementary_slackness:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; init_primal_infeasibility
.. _doxid-structclls__inform__type_init_primal_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_primal_infeasibility

these values at the initial point (needed bg GALAHAD_CCLLS)

.. index:: pair: variable; init_dual_infeasibility
.. _doxid-structclls__inform__type_init_dual_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_dual_infeasibility

see init_primal_infeasibility

.. index:: pair: variable; init_complementary_slackness
.. _doxid-structclls__inform__type_init_complementary_slackness:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T init_complementary_slackness

see init_primal_infeasibility

.. index:: pair: variable; potential
.. _doxid-structclls__inform__type_potential:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structclls__inform__type_non_negligible_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structclls__inform__type_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structclls__inform__type_checkpointsIter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structclls__inform__type_checkpointsTime:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structclls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`clls_time_type<doxid-structclls__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structclls__inform__type_fdc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structclls__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sls_pounce_inform
.. _doxid-structclls__inform__type_sls_pounce_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_pounce_inform

inform parameters for SLS pounce

.. index:: pair: variable; fit_inform
.. _doxid-structclls__inform__type_fit_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

.. index:: pair: variable; roots_inform
.. _doxid-structclls__inform__type_roots_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

return information from ROOTS

.. index:: pair: variable; cro_inform
.. _doxid-structclls__inform__type_cro_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`cro_inform_type<doxid-structcro__inform__type>` cro_inform

inform parameters for CRO

.. index:: pair: variable; rpd_inform
.. _doxid-structclls__inform__type_rpd_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD
