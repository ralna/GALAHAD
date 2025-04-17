.. index:: pair: struct; arc_inform_type
.. _doxid-structarc__inform__type:

arc_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_arc.h>
	
	struct arc_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structarc__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structarc__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structarc__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structarc__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structarc__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structarc__inform__type_f_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structarc__inform__type_g_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structarc__inform__type_h_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structarc__inform__type_factorization_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_max<doxid-structarc__inform__type_factorization_max>`;
		int64_t :ref:`max_entries_factors<doxid-structarc__inform__type_max_entries_factors>`;
		int64_t :ref:`factorization_integer<doxid-structarc__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structarc__inform__type_factorization_real>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorization_average<doxid-structarc__inform__type_factorization_average>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structarc__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_g<doxid-structarc__inform__type_norm_g>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structarc__inform__type_weight>`;
		struct :ref:`arc_time_type<doxid-structarc__time__type>` :ref:`time<doxid-structarc__inform__type_time>`;
		struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` :ref:`rqs_inform<doxid-structarc__inform__type_rqs_inform>`;
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` :ref:`glrt_inform<doxid-structarc__inform__type_glrt_inform>`;
		struct :ref:`dps_inform_type<doxid-structdps__inform__type>` :ref:`dps_inform<doxid-structarc__inform__type_dps_inform>`;
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` :ref:`psls_inform<doxid-structarc__inform__type_psls_inform>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :ref:`lms_inform<doxid-structarc__inform__type_lms_inform>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :target:`lms_inform_prec<doxid-structarc__inform__type_lms_inform_prec>`;
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>` :ref:`sha_inform<doxid-structarc__inform__type_sha_inform>`;
	};
.. _details-structarc__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structarc__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See ARC_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structarc__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structarc__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structarc__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structarc__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of CG iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structarc__inform__type_f_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structarc__inform__type_g_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective functio

.. index:: pair: variable; h_eval
.. _doxid-structarc__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; factorization_status
.. _doxid-structarc__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_max
.. _doxid-structarc__inform__type_factorization_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; max_entries_factors
.. _doxid-structarc__inform__type_max_entries_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structarc__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structarc__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structarc__inform__type_factorization_average:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structarc__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by the package.

.. index:: pair: variable; norm_g
.. _doxid-structarc__inform__type_norm_g:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_g

the norm of the gradient of the objective function at the best estimate of the solution determined by the package.

.. index:: pair: variable; weight
.. _doxid-structarc__inform__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the current value of the regularization weight

.. index:: pair: variable; time
.. _doxid-structarc__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`arc_time_type<doxid-structarc__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structarc__inform__type_rqs_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structarc__inform__type_glrt_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; dps_inform
.. _doxid-structarc__inform__type_dps_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dps_inform_type<doxid-structdps__inform__type>` dps_inform

inform parameters for DPS

.. index:: pair: variable; psls_inform
.. _doxid-structarc__inform__type_psls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; lms_inform
.. _doxid-structarc__inform__type_lms_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform

inform parameters for LMS

.. index:: pair: variable; lms_inform_prec
.. _doxid-structarc__inform__type_lms_inform_prec:
.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform_prec

inform parameters for LMS used for preconditioning

.. index:: pair: variable; sha_inform
.. _doxid-structarc__inform__type_sha_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_inform_type<doxid-structsha__inform__type>` sha_inform

inform parameters for SHA

