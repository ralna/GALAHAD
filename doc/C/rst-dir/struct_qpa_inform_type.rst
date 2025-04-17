.. index:: pair: struct; qpa_inform_type
.. _doxid-structqpa__inform__type:

qpa_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_qpa.h>
	
	struct qpa_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structqpa__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structqpa__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structqpa__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`major_iter<doxid-structqpa__inform__type_major_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structqpa__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structqpa__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structqpa__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structqpa__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structqpa__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structqpa__inform__type_nfacts>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nmods<doxid-structqpa__inform__type_nmods>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`num_g_infeas<doxid-structqpa__inform__type_num_g_infeas>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`num_b_infeas<doxid-structqpa__inform__type_num_b_infeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structqpa__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_g<doxid-structqpa__inform__type_infeas_g>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_b<doxid-structqpa__inform__type_infeas_b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`merit<doxid-structqpa__inform__type_merit>`;
		struct :ref:`qpa_time_type<doxid-structqpa__time__type>` :ref:`time<doxid-structqpa__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structqpa__inform__type_sls_inform>`;
	};
.. _details-structqpa__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpa__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See QPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpa__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpa__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; major_iter
.. _doxid-structqpa__inform__type_major_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` major_iter

the total number of major iterations required

.. index:: pair: variable; iter
.. _doxid-structqpa__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpa__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpa__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpa__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpa__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpa__inform__type_nfacts:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nmods
.. _doxid-structqpa__inform__type_nmods:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; num_g_infeas
.. _doxid-structqpa__inform__type_num_g_infeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` num_g_infeas

the number of infeasible general constraints

.. index:: pair: variable; num_b_infeas
.. _doxid-structqpa__inform__type_num_b_infeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` num_b_infeas

the number of infeasible simple-bound constraints

.. index:: pair: variable; obj
.. _doxid-structqpa__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by QPA_solve

.. index:: pair: variable; infeas_g
.. _doxid-structqpa__inform__type_infeas_g:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_g

the 1-norm of the infeasibility of the general constraints

.. index:: pair: variable; infeas_b
.. _doxid-structqpa__inform__type_infeas_b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_b

the 1-norm of the infeasibility of the simple-bound constraints

.. index:: pair: variable; merit
.. _doxid-structqpa__inform__type_merit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` merit

the merit function value = obj + rho_g \* infeas_g + rho_b \* infeas_b

.. index:: pair: variable; time
.. _doxid-structqpa__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`qpa_time_type<doxid-structqpa__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structqpa__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

