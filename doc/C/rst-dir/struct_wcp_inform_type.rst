.. index:: pair: struct; wcp_inform_type
.. _doxid-structwcp__inform__type:

wcp_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_wcp.h>
	
	struct wcp_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structwcp__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structwcp__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structwcp__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structwcp__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structwcp__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structwcp__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structwcp__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structwcp__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`c_implicit<doxid-structwcp__inform__type_c_implicit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`x_implicit<doxid-structwcp__inform__type_x_implicit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`y_implicit<doxid-structwcp__inform__type_y_implicit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`z_implicit<doxid-structwcp__inform__type_z_implicit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structwcp__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_final_target_max<doxid-structwcp__inform__type_mu_final_target_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structwcp__inform__type_non_negligible_pivot>`;
		bool :ref:`feasible<doxid-structwcp__inform__type_feasible>`;
		struct :ref:`wcp_time_type<doxid-structwcp__time__type>` :ref:`time<doxid-structwcp__inform__type_time>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structwcp__inform__type_fdc_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structwcp__inform__type_sbls_inform>`;
	};
.. _details-structwcp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structwcp__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See WCP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structwcp__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structwcp__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structwcp__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structwcp__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structwcp__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structwcp__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structwcp__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; c_implicit
.. _doxid-structwcp__inform__type_c_implicit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` c_implicit

the number of general constraints that lie on (one) of their bounds for feasible solutions

.. index:: pair: variable; x_implicit
.. _doxid-structwcp__inform__type_x_implicit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` x_implicit

the number of variables that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; y_implicit
.. _doxid-structwcp__inform__type_y_implicit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` y_implicit

the number of Lagrange multipliers for general constraints that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; z_implicit
.. _doxid-structwcp__inform__type_z_implicit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` z_implicit

the number of dual variables that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; obj
.. _doxid-structwcp__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by WCP_solve

.. index:: pair: variable; mu_final_target_max
.. _doxid-structwcp__inform__type_mu_final_target_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_final_target_max

the largest target value on termination

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structwcp__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structwcp__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned primal-dual "solution" strictly feasible?

.. index:: pair: variable; time
.. _doxid-structwcp__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`wcp_time_type<doxid-structwcp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structwcp__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structwcp__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

