.. index:: pair: struct; dqp_inform_type
.. _doxid-structdqp__inform__type:

dqp_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dqp.h>
	
	struct dqp_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structdqp__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structdqp__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structdqp__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structdqp__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structdqp__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structdqp__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structdqp__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structdqp__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structdqp__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`threads<doxid-structdqp__inform__type_threads>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structdqp__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structdqp__inform__type_primal_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dual_infeasibility<doxid-structdqp__inform__type_dual_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`complementary_slackness<doxid-structdqp__inform__type_complementary_slackness>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structdqp__inform__type_non_negligible_pivot>`;
		bool :ref:`feasible<doxid-structdqp__inform__type_feasible>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`checkpointsIter<doxid-structdqp__inform__type_checkpointsIter>`[16];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`checkpointsTime<doxid-structdqp__inform__type_checkpointsTime>`[16];
		struct :ref:`dqp_time_type<doxid-structdqp__time__type>` :ref:`time<doxid-structdqp__inform__type_time>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structdqp__inform__type_fdc_inform>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structdqp__inform__type_sls_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structdqp__inform__type_sbls_inform>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structdqp__inform__type_gltr_inform>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scu_status<doxid-structdqp__inform__type_scu_status>`;
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>` :ref:`scu_inform<doxid-structdqp__inform__type_scu_inform>`;
		struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` :ref:`rpd_inform<doxid-structdqp__inform__type_rpd_inform>`;
	};
.. _details-structdqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdqp__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See DQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdqp__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structdqp__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structdqp__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structdqp__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structdqp__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structdqp__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structdqp__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structdqp__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; threads
.. _doxid-structdqp__inform__type_threads:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structdqp__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by DQP_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structdqp__inform__type_primal_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structdqp__inform__type_dual_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structdqp__inform__type_complementary_slackness:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structdqp__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot that was not judged to be zero when detecting linearly dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structdqp__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structdqp__inform__type_checkpointsIter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structdqp__inform__type_checkpointsTime:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structdqp__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dqp_time_type<doxid-structdqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structdqp__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structdqp__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structdqp__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structdqp__inform__type_gltr_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; scu_status
.. _doxid-structdqp__inform__type_scu_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scu_status

inform parameters for SCU

.. index:: pair: variable; scu_inform
.. _doxid-structdqp__inform__type_scu_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; rpd_inform
.. _doxid-structdqp__inform__type_rpd_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

