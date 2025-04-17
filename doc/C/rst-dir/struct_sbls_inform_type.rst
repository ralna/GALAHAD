.. index:: pair: table; sbls_inform_type
.. _doxid-structsbls__inform__type:

sbls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sbls.h>
	
	struct sbls_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structsbls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structsbls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structsbls__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sort_status<doxid-structsbls__inform__type_sort_status>`;
		int64_t :ref:`factorization_integer<doxid-structsbls__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structsbls__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structsbls__inform__type_preconditioner>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization<doxid-structsbls__inform__type_factorization>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`d_plus<doxid-structsbls__inform__type_d_plus>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structsbls__inform__type_rank>`;
		bool :ref:`rank_def<doxid-structsbls__inform__type_rank_def>`;
		bool :ref:`perturbed<doxid-structsbls__inform__type_perturbed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pcg<doxid-structsbls__inform__type_iter_pcg>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_residual<doxid-structsbls__inform__type_norm_residual>`;
		bool :ref:`alternative<doxid-structsbls__inform__type_alternative>`;
		struct :ref:`sbls_time_type<doxid-structsbls__time__type>` :ref:`time<doxid-structsbls__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structsbls__inform__type_sls_inform>`;
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>` :ref:`uls_inform<doxid-structsbls__inform__type_uls_inform>`;
	};
.. _details-structsbls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsbls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See SBLS_form_and_factorize for details

.. index:: pair: variable; alloc_status
.. _doxid-structsbls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structsbls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; sort_status
.. _doxid-structsbls__inform__type_sort_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sort_status

the return status from the sorting routines

.. index:: pair: variable; factorization_integer
.. _doxid-structsbls__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structsbls__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; preconditioner
.. _doxid-structsbls__inform__type_preconditioner:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

the preconditioner used

.. index:: pair: variable; factorization
.. _doxid-structsbls__inform__type_factorization:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization

the factorization used

.. index:: pair: variable; d_plus
.. _doxid-structsbls__inform__type_d_plus:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` d_plus

how many of the diagonals in the factorization are positive

.. index:: pair: variable; rank
.. _doxid-structsbls__inform__type_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

the computed rank of $A$

.. index:: pair: variable; rank_def
.. _doxid-structsbls__inform__type_rank_def:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool rank_def

is the matrix A rank defficient?

.. index:: pair: variable; perturbed
.. _doxid-structsbls__inform__type_perturbed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool perturbed

has the used preconditioner been perturbed to guarantee correct inertia?

.. index:: pair: variable; iter_pcg
.. _doxid-structsbls__inform__type_iter_pcg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pcg

the total number of projected CG iterations required

.. index:: pair: variable; norm_residual
.. _doxid-structsbls__inform__type_norm_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_residual

the norm of the residual

.. index:: pair: variable; alternative
.. _doxid-structsbls__inform__type_alternative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool alternative

has an "alternative" $y$ : $K y = 0$ and $y^T c > 0$ been found when trying to solve $K y = c$ for generic $K$?

.. index:: pair: variable; time
.. _doxid-structsbls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_time_type<doxid-structsbls__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structsbls__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters from the GALAHAD package SLS used

.. index:: pair: variable; uls_inform
.. _doxid-structsbls__inform__type_uls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

inform parameters from the GALAHAD package ULS used

