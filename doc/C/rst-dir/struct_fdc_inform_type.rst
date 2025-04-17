.. index:: pair: table; fdc_inform_type
.. _doxid-structfdc__inform__type:

fdc_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_fdc.h>
	
	struct fdc_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structfdc__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structfdc__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structfdc__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structfdc__inform__type_factorization_status>`;
		int64_t :ref:`factorization_integer<doxid-structfdc__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structfdc__inform__type_factorization_real>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structfdc__inform__type_non_negligible_pivot>`;
		struct :ref:`fdc_time_type<doxid-structfdc__time__type>` :ref:`time<doxid-structfdc__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structfdc__inform__type_sls_inform>`;
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>` :ref:`uls_inform<doxid-structfdc__inform__type_uls_inform>`;
	};
.. _details-structfdc__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structfdc__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See FDC_find_dependent for details

.. index:: pair: variable; alloc_status
.. _doxid-structfdc__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structfdc__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; factorization_status
.. _doxid-structfdc__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structfdc__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structfdc__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structfdc__inform__type_non_negligible_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; time
.. _doxid-structfdc__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_time_type<doxid-structfdc__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structfdc__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

SLS inform type.

.. index:: pair: variable; uls_inform
.. _doxid-structfdc__inform__type_uls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

ULS inform type.

