.. index:: pair: table; ssls_inform_type
.. _doxid-structssls__inform__type:

ssls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ssls.h>
	
	struct ssls_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structssls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structssls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structssls__inform__type_bad_alloc>`[81];
		int64_t :ref:`factorization_integer<doxid-structssls__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structssls__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structssls__inform__type_rank>`;
		bool :ref:`rank_def<doxid-structssls__inform__type_rank_def>`;
		struct :ref:`ssls_time_type<doxid-structssls__time__type>` :ref:`time<doxid-structssls__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structssls__inform__type_sls_inform>`;
	};
.. _details-structssls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structssls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See SSLS_form_and_factorize for details

.. index:: pair: variable; alloc_status
.. _doxid-structssls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structssls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; factorization_integer
.. _doxid-structssls__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structssls__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; rank
.. _doxid-structssls__inform__type_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

the computed rank of $A$

.. index:: pair: variable; rank_def
.. _doxid-structssls__inform__type_rank_def:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool rank_def

is the matrix A rank defficient?

.. index:: pair: variable; time
.. _doxid-structssls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ssls_time_type<doxid-structssls__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structssls__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters from the GALAHAD package SLS used
