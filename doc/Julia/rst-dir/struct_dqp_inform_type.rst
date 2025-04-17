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
.. _doxid-structdqp__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See DQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdqp__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structdqp__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structdqp__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structdqp__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structdqp__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structdqp__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structdqp__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structdqp__inform__type_nfacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; threads
.. _doxid-structdqp__inform__type_threads:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structdqp__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by DQP_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structdqp__inform__type_primal_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structdqp__inform__type_dual_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structdqp__inform__type_complementary_slackness:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structdqp__inform__type_non_negligible_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot that was not judged to be zero when detecting linearly dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structdqp__inform__type_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structdqp__inform__type_checkpointsIter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structdqp__inform__type_checkpointsTime:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structdqp__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dqp_time_type<doxid-structdqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structdqp__inform__type_fdc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structdqp__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structdqp__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structdqp__inform__type_gltr_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; scu_status
.. _doxid-structdqp__inform__type_scu_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT scu_status

inform parameters for SCU

.. index:: pair: variable; scu_inform
.. _doxid-structdqp__inform__type_scu_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; rpd_inform
.. _doxid-structdqp__inform__type_rpd_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

