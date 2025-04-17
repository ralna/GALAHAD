.. index:: pair: table; psls_inform_type
.. _doxid-structpsls__inform__type:

psls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_psls.h>
	
	struct psls_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structpsls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structpsls__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`analyse_status<doxid-structpsls__inform__type_analyse_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorize_status<doxid-structpsls__inform__type_factorize_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`solve_status<doxid-structpsls__inform__type_solve_status>`;
		int64_t :ref:`factorization_integer<doxid-structpsls__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structpsls__inform__type_factorization_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structpsls__inform__type_preconditioner>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structpsls__inform__type_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`reordered_semi_bandwidth<doxid-structpsls__inform__type_reordered_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out_of_range<doxid-structpsls__inform__type_out_of_range>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`duplicates<doxid-structpsls__inform__type_duplicates>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`upper<doxid-structpsls__inform__type_upper>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`missing_diagonals<doxid-structpsls__inform__type_missing_diagonals>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth_used<doxid-structpsls__inform__type_semi_bandwidth_used>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`neg1<doxid-structpsls__inform__type_neg1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`neg2<doxid-structpsls__inform__type_neg2>`;
		bool :ref:`perturbed<doxid-structpsls__inform__type_perturbed>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fill_in_ratio<doxid-structpsls__inform__type_fill_in_ratio>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_residual<doxid-structpsls__inform__type_norm_residual>`;
		char :ref:`bad_alloc<doxid-structpsls__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mc61_info<doxid-structpsls__inform__type_mc61_info>`[10];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mc61_rinfo<doxid-structpsls__inform__type_mc61_rinfo>`[15];
		struct :ref:`psls_time_type<doxid-structpsls__time__type>` :ref:`time<doxid-structpsls__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structpsls__inform__type_sls_inform>`;
		struct :ref:`mi28_info<doxid-structmi28__info>` :ref:`mi28_info<doxid-structpsls__inform__type_mi28_info>`;
	};
.. _details-structpsls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structpsls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status:

* **0**

  success

* **-1**

  allocation error

* **-2**

  deallocation error

* **-3**

  matrix data faulty (.n < 1, .ne < 0)

.. index:: pair: variable; alloc_status
.. _doxid-structpsls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; analyse_status
.. _doxid-structpsls__inform__type_analyse_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` analyse_status

status return from factorization

.. index:: pair: variable; factorize_status
.. _doxid-structpsls__inform__type_factorize_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorize_status

status return from factorization

.. index:: pair: variable; solve_status
.. _doxid-structpsls__inform__type_solve_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` solve_status

status return from solution phase

.. index:: pair: variable; factorization_integer
.. _doxid-structpsls__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

number of integer words to hold factors

.. index:: pair: variable; factorization_real
.. _doxid-structpsls__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

number of real words to hold factors

.. index:: pair: variable; preconditioner
.. _doxid-structpsls__inform__type_preconditioner:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

code for the actual preconditioner used (see control.preconditioner)

.. index:: pair: variable; semi_bandwidth
.. _doxid-structpsls__inform__type_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

the actual semi-bandwidth

.. index:: pair: variable; reordered_semi_bandwidth
.. _doxid-structpsls__inform__type_reordered_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` reordered_semi_bandwidth

the semi-bandwidth following reordering (if any)

.. index:: pair: variable; out_of_range
.. _doxid-structpsls__inform__type_out_of_range:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structpsls__inform__type_duplicates:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structpsls__inform__type_upper:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structpsls__inform__type_missing_diagonals:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; semi_bandwidth_used
.. _doxid-structpsls__inform__type_semi_bandwidth_used:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth_used

the semi-bandwidth used

.. index:: pair: variable; neg1
.. _doxid-structpsls__inform__type_neg1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` neg1

number of 1 by 1 pivots in the factorization

.. index:: pair: variable; neg2
.. _doxid-structpsls__inform__type_neg2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` neg2

number of 2 by 2 pivots in the factorization

.. index:: pair: variable; perturbed
.. _doxid-structpsls__inform__type_perturbed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool perturbed

has the preconditioner been perturbed during the fctorization?

.. index:: pair: variable; fill_in_ratio
.. _doxid-structpsls__inform__type_fill_in_ratio:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fill_in_ratio

ratio of fill in to original nonzeros

.. index:: pair: variable; norm_residual
.. _doxid-structpsls__inform__type_norm_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_residual

the norm of the solution residual

.. index:: pair: variable; bad_alloc
.. _doxid-structpsls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; mc61_info
.. _doxid-structpsls__inform__type_mc61_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mc61_info[10]

the integer and real output arrays from mc61

.. index:: pair: variable; mc61_rinfo
.. _doxid-structpsls__inform__type_mc61_rinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mc61_rinfo[15]

see mc61_info

.. index:: pair: variable; time
.. _doxid-structpsls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_time_type<doxid-structpsls__time__type>` time

times for various stages

.. index:: pair: variable; sls_inform
.. _doxid-structpsls__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform values from SLS

.. index:: pair: variable; mi28_info
.. _doxid-structpsls__inform__type_mi28_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`mi28_info<doxid-structmi28__info>` mi28_info

the output info structure from HSL's mi28

