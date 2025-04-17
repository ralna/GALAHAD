.. index:: pair: table; lstr_inform_type
.. _doxid-structlstr__inform__type:

lstr_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lstr.h>
	
	struct lstr_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlstr__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlstr__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structlstr__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structlstr__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pass2<doxid-structlstr__inform__type_iter_pass2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biters<doxid-structlstr__inform__type_biters>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_min<doxid-structlstr__inform__type_biter_min>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_max<doxid-structlstr__inform__type_biter_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structlstr__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structlstr__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structlstr__inform__type_r_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`Atr_norm<doxid-structlstr__inform__type_Atr_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`biter_mean<doxid-structlstr__inform__type_biter_mean>`;
	};
.. _details-structlstr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlstr__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See :ref:`lstr_solve_problem <doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structlstr__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlstr__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlstr__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structlstr__inform__type_iter_pass2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pass2

the total number of pass-2 iterations required if the solution lies on the trust-region boundary

.. index:: pair: variable; biters
.. _doxid-structlstr__inform__type_biters:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structlstr__inform__type_biter_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structlstr__inform__type_biter_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_max

the largestt number of inner iterations performed during an outer iteration

.. index:: pair: variable; multiplier
.. _doxid-structlstr__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier, $\lambda$, corresponding to the trust-region constraint

.. index:: pair: variable; x_norm
.. _doxid-structlstr__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the Euclidean norm of $x$

.. index:: pair: variable; r_norm
.. _doxid-structlstr__inform__type_r_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

the Euclidean norm of $Ax-b$

.. index:: pair: variable; Atr_norm
.. _doxid-structlstr__inform__type_Atr_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` Atr_norm

the Euclidean norm of $A^T (Ax-b) + \lambda x$

.. index:: pair: variable; biter_mean
.. _doxid-structlstr__inform__type_biter_mean:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` biter_mean

the average number of inner iterations performed during an outer

iteration

