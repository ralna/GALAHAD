.. index:: pair: struct; lsqp_inform_type
.. _doxid-structlsqp__inform__type:

lsqp_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lsqp.h>
	
	struct lsqp_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlsqp__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlsqp__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structlsqp__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structlsqp__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structlsqp__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structlsqp__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structlsqp__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structlsqp__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbacts<doxid-structlsqp__inform__type_nbacts>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structlsqp__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`potential<doxid-structlsqp__inform__type_potential>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structlsqp__inform__type_non_negligible_pivot>`;
		bool :ref:`feasible<doxid-structlsqp__inform__type_feasible>`;
		struct :ref:`lsqp_time_type<doxid-structlsqp__time__type>` :ref:`time<doxid-structlsqp__inform__type_time>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structlsqp__inform__type_fdc_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structlsqp__inform__type_sbls_inform>`;
	};
.. _details-structlsqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlsqp__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See LSQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlsqp__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlsqp__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlsqp__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structlsqp__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structlsqp__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structlsqp__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structlsqp__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structlsqp__inform__type_nbacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; obj
.. _doxid-structlsqp__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by LSQP_solve_qp

.. index:: pair: variable; potential
.. _doxid-structlsqp__inform__type_potential:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structlsqp__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structlsqp__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structlsqp__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_time_type<doxid-structlsqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structlsqp__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structlsqp__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

