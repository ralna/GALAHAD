.. index:: pair: table; l2rt_inform_type
.. _doxid-structl2rt__inform__type:

l2rt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_l2rt.h>
	
	struct l2rt_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structl2rt__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structl2rt__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structl2rt__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structl2rt__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pass2<doxid-structl2rt__inform__type_iter_pass2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biters<doxid-structl2rt__inform__type_biters>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_min<doxid-structl2rt__inform__type_biter_min>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_max<doxid-structl2rt__inform__type_biter_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structl2rt__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structl2rt__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structl2rt__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structl2rt__inform__type_r_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`Atr_norm<doxid-structl2rt__inform__type_Atr_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`biter_mean<doxid-structl2rt__inform__type_biter_mean>`;
	};
.. _details-structl2rt__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structl2rt__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See :ref:`l2rt_solve_problem <doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structl2rt__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structl2rt__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structl2rt__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structl2rt__inform__type_iter_pass2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; biters
.. _doxid-structl2rt__inform__type_biters:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structl2rt__inform__type_biter_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structl2rt__inform__type_biter_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_max

the largest number of inner iterations performed during an outer iteration

.. index:: pair: variable; obj
.. _doxid-structl2rt__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function

.. index:: pair: variable; multiplier
.. _doxid-structl2rt__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the multiplier, $\lambda = \mu + \sigma \|x\|^{p-2} * \sqrt{\|Ax-b\|^2 + \mu \|x\|^2}$

.. index:: pair: variable; x_norm
.. _doxid-structl2rt__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the Euclidean norm of $x$

.. index:: pair: variable; r_norm
.. _doxid-structl2rt__inform__type_r_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

the Euclidean norm of $Ax-b$

.. index:: pair: variable; Atr_norm
.. _doxid-structl2rt__inform__type_Atr_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` Atr_norm

the Euclidean norm of $A^T (Ax-b) + \lambda x$

.. index:: pair: variable; biter_mean
.. _doxid-structl2rt__inform__type_biter_mean:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` biter_mean

the average number of inner iterations performed during an outer iteration

