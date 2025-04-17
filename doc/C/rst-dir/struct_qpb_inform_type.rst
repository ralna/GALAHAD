.. index:: pair: struct; qpb_inform_type
.. _doxid-structqpb__inform__type:

qpb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_qpb.h>
	
	struct qpb_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structqpb__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structqpb__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structqpb__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structqpb__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structqpb__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structqpb__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structqpb__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structqpb__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structqpb__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbacts<doxid-structqpb__inform__type_nbacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nmods<doxid-structqpb__inform__type_nmods>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structqpb__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structqpb__inform__type_non_negligible_pivot>`;
		bool :ref:`feasible<doxid-structqpb__inform__type_feasible>`;
		struct :ref:`qpb_time_type<doxid-structqpb__time__type>` :ref:`time<doxid-structqpb__inform__type_time>`;
		struct :ref:`lsqp_inform_type<doxid-structlsqp__inform__type>` :ref:`lsqp_inform<doxid-structqpb__inform__type_lsqp_inform>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structqpb__inform__type_fdc_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structqpb__inform__type_sbls_inform>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structqpb__inform__type_gltr_inform>`;
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>` :ref:`fit_inform<doxid-structqpb__inform__type_fit_inform>`;
	};
.. _details-structqpb__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpb__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See QPB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpb__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpb__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structqpb__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpb__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpb__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpb__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpb__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpb__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structqpb__inform__type_nbacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; nmods
.. _doxid-structqpb__inform__type_nmods:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; obj
.. _doxid-structqpb__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structqpb__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structqpb__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structqpb__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`qpb_time_type<doxid-structqpb__time__type>` time

timings (see above)

.. index:: pair: variable; lsqp_inform
.. _doxid-structqpb__inform__type_lsqp_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_inform_type<doxid-structlsqp__inform__type>` lsqp_inform

inform parameters for LSQP

.. index:: pair: variable; fdc_inform
.. _doxid-structqpb__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structqpb__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structqpb__inform__type_gltr_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; fit_inform
.. _doxid-structqpb__inform__type_fit_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

