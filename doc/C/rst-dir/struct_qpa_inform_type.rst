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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structqpa__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structqpa__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structqpa__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`major_iter<doxid-structqpa__inform__type_1a17745a543d52bf415bce3e518de5c244>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structqpa__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structqpa__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structqpa__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`factorization_integer<doxid-structqpa__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structqpa__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structqpa__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nmods<doxid-structqpa__inform__type_1aedcbf93d59a135329f358f366e37cc94>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`num_g_infeas<doxid-structqpa__inform__type_1ad05d7e095223165473fa80bd34520fb4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`num_b_infeas<doxid-structqpa__inform__type_1a9ac73703449a79c042833709b93cbba5>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structqpa__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_g<doxid-structqpa__inform__type_1a3e76c2cdb1a38096c0ab34782ac6497b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_b<doxid-structqpa__inform__type_1a3001ce44f7449e075ae83fde8439c8df>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`merit<doxid-structqpa__inform__type_1a231e0f500c3a8e0e9acfa786b03381d9>`;
		struct :ref:`qpa_time_type<doxid-structqpa__time__type>` :ref:`time<doxid-structqpa__inform__type_1af9a10dd74244e6c12e136ae0828ae3a7>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structqpa__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`;
	};
.. _details-structqpa__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpa__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See QPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpa__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpa__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; major_iter
.. _doxid-structqpa__inform__type_1a17745a543d52bf415bce3e518de5c244:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` major_iter

the total number of major iterations required

.. index:: pair: variable; iter
.. _doxid-structqpa__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpa__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpa__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpa__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpa__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpa__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nmods
.. _doxid-structqpa__inform__type_1aedcbf93d59a135329f358f366e37cc94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; num_g_infeas
.. _doxid-structqpa__inform__type_1ad05d7e095223165473fa80bd34520fb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` num_g_infeas

the number of infeasible general constraints

.. index:: pair: variable; num_b_infeas
.. _doxid-structqpa__inform__type_1a9ac73703449a79c042833709b93cbba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` num_b_infeas

the number of infeasible simple-bound constraints

.. index:: pair: variable; obj
.. _doxid-structqpa__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by QPA_solve

.. index:: pair: variable; infeas_g
.. _doxid-structqpa__inform__type_1a3e76c2cdb1a38096c0ab34782ac6497b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_g

the 1-norm of the infeasibility of the general constraints

.. index:: pair: variable; infeas_b
.. _doxid-structqpa__inform__type_1a3001ce44f7449e075ae83fde8439c8df:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_b

the 1-norm of the infeasibility of the simple-bound constraints

.. index:: pair: variable; merit
.. _doxid-structqpa__inform__type_1a231e0f500c3a8e0e9acfa786b03381d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` merit

the merit function value = obj + rho_g \* infeas_g + rho_b \* infeas_b

.. index:: pair: variable; time
.. _doxid-structqpa__inform__type_1af9a10dd74244e6c12e136ae0828ae3a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`qpa_time_type<doxid-structqpa__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structqpa__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

