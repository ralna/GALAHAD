.. index:: pair: struct; lpa_inform_type
.. _doxid-structlpa__inform__type:

lpa_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lpa.h>
	
	struct lpa_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlpa__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlpa__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structlpa__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structlpa__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la04_job<doxid-structlpa__inform__type_la04_job>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la04_job_info<doxid-structlpa__inform__type_la04_job_info>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structlpa__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structlpa__inform__type_primal_infeasibility>`;
		bool :ref:`feasible<doxid-structlpa__inform__type_feasible>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`RINFO<doxid-structlpa__inform__type_RINFO>`[40];
		struct :ref:`lpa_time_type<doxid-structlpa__time__type>` :ref:`time<doxid-structlpa__inform__type_time>`;
		struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` :ref:`rpd_inform<doxid-structlpa__inform__type_rpd_inform>`;
	};
.. _details-structlpa__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlpa__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See LPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlpa__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlpa__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlpa__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; la04_job
.. _doxid-structlpa__inform__type_la04_job:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la04_job

the final value of la04's job argument

.. index:: pair: variable; la04_job_info
.. _doxid-structlpa__inform__type_la04_job_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la04_job_info

any extra information from an unsuccessfull call to LA04 (LA04's RINFO(35)

.. index:: pair: variable; obj
.. _doxid-structlpa__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by LPA_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structlpa__inform__type_primal_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; feasible
.. _doxid-structlpa__inform__type_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; RINFO
.. _doxid-structlpa__inform__type_RINFO:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` RINFO[40]

the information array from LA04

.. index:: pair: variable; time
.. _doxid-structlpa__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lpa_time_type<doxid-structlpa__time__type>` time

timings (see above)

.. index:: pair: variable; rpd_inform
.. _doxid-structlpa__inform__type_rpd_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

