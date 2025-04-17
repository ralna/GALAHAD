.. index:: pair: struct; eqp_inform_type
.. _doxid-structeqp__inform__type:

eqp_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_eqp.h>
	
	struct eqp_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structeqp__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structeqp__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structeqp__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structeqp__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter_inter<doxid-structeqp__inform__type_cg_iter_inter>`;
		int64_t :ref:`factorization_integer<doxid-structeqp__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structeqp__inform__type_factorization_real>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structeqp__inform__type_obj>`;
		struct :ref:`eqp_time_type<doxid-structeqp__time__type>` :ref:`time<doxid-structeqp__inform__type_time>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structeqp__inform__type_fdc_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structeqp__inform__type_sbls_inform>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structeqp__inform__type_gltr_inform>`;
	};
.. _details-structeqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structeqp__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See EQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structeqp__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structeqp__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; cg_iter
.. _doxid-structeqp__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; cg_iter_inter
.. _doxid-structeqp__inform__type_cg_iter_inter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter_inter

see cg_iter

.. index:: pair: variable; factorization_integer
.. _doxid-structeqp__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structeqp__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structeqp__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; time
.. _doxid-structeqp__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`eqp_time_type<doxid-structeqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structeqp__inform__type_fdc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structeqp__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structeqp__inform__type_gltr_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

