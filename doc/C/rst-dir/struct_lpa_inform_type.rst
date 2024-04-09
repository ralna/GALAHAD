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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlpa__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlpa__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structlpa__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structlpa__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la04_job<doxid-structlpa__inform__type_1a8ba753c55f7e33211718d8f58ccfdea3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la04_job_info<doxid-structlpa__inform__type_1acfd9252cb6fa18ef44baaeaab705d85f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structlpa__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structlpa__inform__type_1a2bce6cd733ae08834689fa66747f53b9>`;
		bool :ref:`feasible<doxid-structlpa__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`RINFO<doxid-structlpa__inform__type_1a1dcb2a53d683485290d30e0a16d7e2ee>`[40];
		struct :ref:`lpa_time_type<doxid-structlpa__time__type>` :ref:`time<doxid-structlpa__inform__type_1a06efd7a01012eda3b046d741ef9584fa>`;
		struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` :ref:`rpd_inform<doxid-structlpa__inform__type_1a823701505feea7615e9f8995769d8b60>`;
	};
.. _details-structlpa__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlpa__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See LPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlpa__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlpa__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlpa__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; la04_job
.. _doxid-structlpa__inform__type_1a8ba753c55f7e33211718d8f58ccfdea3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la04_job

the final value of la04's job argument

.. index:: pair: variable; la04_job_info
.. _doxid-structlpa__inform__type_1acfd9252cb6fa18ef44baaeaab705d85f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la04_job_info

any extra information from an unsuccessfull call to LA04 (LA04's RINFO(35)

.. index:: pair: variable; obj
.. _doxid-structlpa__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by LPA_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structlpa__inform__type_1a2bce6cd733ae08834689fa66747f53b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; feasible
.. _doxid-structlpa__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; RINFO
.. _doxid-structlpa__inform__type_1a1dcb2a53d683485290d30e0a16d7e2ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` RINFO[40]

the information array from LA04

.. index:: pair: variable; time
.. _doxid-structlpa__inform__type_1a06efd7a01012eda3b046d741ef9584fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lpa_time_type<doxid-structlpa__time__type>` time

timings (see above)

.. index:: pair: variable; rpd_inform
.. _doxid-structlpa__inform__type_1a823701505feea7615e9f8995769d8b60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

