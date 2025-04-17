.. index:: pair: struct; clls_inform_type
.. _doxid-structclls__inform__type:

clls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_clls.h>
	
	struct clls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structclls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structclls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structclls__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structclls__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structclls__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structclls__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structclls__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structclls__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbacts<doxid-structclls__inform__type_nbacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`threads<doxid-structclls__inform__type_threads>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structclls__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structclls__inform__type_primal_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dual_infeasibility<doxid-structclls__inform__type_dual_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`complementary_slackness<doxid-structclls__inform__type_complementary_slackness>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_primal_infeasibility<doxid-structclls__inform__type_init_primal_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_dual_infeasibility<doxid-structclls__inform__type_init_dual_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_complementary_slackness<doxid-structclls__inform__type_init_complementary_slackness>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`potential<doxid-structclls__inform__type_potential>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structclls__inform__type_non_negligible_pivot>`;
		bool :ref:`feasible<doxid-structclls__inform__type_feasible>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`checkpointsIter<doxid-structclls__inform__type_checkpointsIter>`[16];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`checkpointsTime<doxid-structclls__inform__type_checkpointsTime>`[16];
		struct :ref:`clls_time_type<doxid-structclls__time__type>` :ref:`time<doxid-structclls__inform__type_time>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structclls__inform__type_fdc_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structclls__inform__type_sbls_inform>`;
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>` :ref:`fit_inform<doxid-structclls__inform__type_fit_inform>`;
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>` :ref:`roots_inform<doxid-structclls__inform__type_roots_inform>`;
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>` :ref:`cro_inform<doxid-structclls__inform__type_cro_inform>`;
		struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` :ref:`rpd_inform<doxid-structclls__inform__type_rpd_inform>`;
	};
.. _details-structclls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structclls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See CLLS_solve for details.

.. index:: pair: variable; alloc_status
.. _doxid-structclls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structclls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; iter
.. _doxid-structclls__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structclls__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structclls__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structclls__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structclls__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structclls__inform__type_nbacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; threads
.. _doxid-structclls__inform__type_threads:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structclls__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by CLLS_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structclls__inform__type_primal_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structclls__inform__type_dual_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structclls__inform__type_complementary_slackness:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structclls__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structclls__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structclls__inform__type_checkpointsIter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structclls__inform__type_checkpointsTime:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structclls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`clls_time_type<doxid-structclls__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structclls__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structclls__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsbls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sls_pounce_inform
.. _doxid-structclls__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsbls__inform__type>` sls_pounce_inform

inform parameters for SLS_pounce
.. index:: pair: variable; fit_inform
.. _doxid-structclls__inform__type_fit_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

.. index:: pair: variable; roots_inform
.. _doxid-structclls__inform__type_roots_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

return information from ROOTS

.. index:: pair: variable; cro_inform
.. _doxid-structclls__inform__type_cro_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`cro_inform_type<doxid-structcro__inform__type>` cro_inform

inform parameters for CRO

.. index:: pair: variable; rpd_inform
.. _doxid-structclls__inform__type_rpd_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

