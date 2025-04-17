.. index:: pair: struct; trb_inform_type
.. _doxid-structtrb__inform__type:

trb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trb.h>
	
	struct trb_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structtrb__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structtrb__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structtrb__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structtrb__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structtrb__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structtrb__inform__type_cg_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structtrb__inform__type_f_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structtrb__inform__type_g_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structtrb__inform__type_h_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`n_free<doxid-structtrb__inform__type_n_free>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_max<doxid-structtrb__inform__type_factorization_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structtrb__inform__type_factorization_status>`;
		int64_t :ref:`max_entries_factors<doxid-structtrb__inform__type_max_entries_factors>`;
		int64_t :ref:`factorization_integer<doxid-structtrb__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structtrb__inform__type_factorization_real>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structtrb__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structtrb__inform__type_norm_pg>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius<doxid-structtrb__inform__type_radius>`;
		struct :ref:`trb_time_type<doxid-structtrb__time__type>` :ref:`time<doxid-structtrb__inform__type_time>`;
		struct :ref:`trs_inform_type<doxid-structtrs__inform__type>` :ref:`trs_inform<doxid-structtrb__inform__type_trs_inform>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structtrb__inform__type_gltr_inform>`;
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` :ref:`psls_inform<doxid-structtrb__inform__type_psls_inform>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :ref:`lms_inform<doxid-structtrb__inform__type_lms_inform>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :ref:`lms_inform_prec<doxid-structtrb__inform__type_lms_inform_prec>`;
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>` :ref:`sha_inform<doxid-structtrb__inform__type_sha_inform>`;
	};
.. _details-structtrb__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structtrb__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See TRB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structtrb__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structtrb__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structtrb__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structtrb__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of CG iterations performed

.. index:: pair: variable; cg_maxit
.. _doxid-structtrb__inform__type_cg_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

the maximum number of CG iterations allowed per iteration

.. index:: pair: variable; f_eval
.. _doxid-structtrb__inform__type_f_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structtrb__inform__type_g_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structtrb__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; n_free
.. _doxid-structtrb__inform__type_n_free:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` n_free

the number of variables that are free from their bounds

.. index:: pair: variable; factorization_max
.. _doxid-structtrb__inform__type_factorization_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structtrb__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structtrb__inform__type_max_entries_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structtrb__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structtrb__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structtrb__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; norm_pg
.. _doxid-structtrb__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; radius
.. _doxid-structtrb__inform__type_radius:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius

the current value of the trust-region radius

.. index:: pair: variable; time
.. _doxid-structtrb__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_time_type<doxid-structtrb__time__type>` time

timings (see above)

.. index:: pair: variable; trs_inform
.. _doxid-structtrb__inform__type_trs_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_inform_type<doxid-structtrs__inform__type>` trs_inform

inform parameters for TRS

.. index:: pair: variable; gltr_inform
.. _doxid-structtrb__inform__type_gltr_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

inform parameters for GLTR

.. index:: pair: variable; psls_inform
.. _doxid-structtrb__inform__type_psls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; lms_inform
.. _doxid-structtrb__inform__type_lms_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform

inform parameters for LMS

.. index:: pair: variable; lms_inform_prec
.. _doxid-structtrb__inform__type_lms_inform_prec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform_prec

inform parameters for LMS used for preconditioning

.. index:: pair: variable; sha_inform
.. _doxid-structtrb__inform__type_sha_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_inform_type<doxid-structsha__inform__type>` sha_inform

inform parameters for SHA

