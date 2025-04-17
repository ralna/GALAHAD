.. index:: pair: table; glrt_inform_type
.. _doxid-structglrt__inform__type:

glrt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_glrt.h>
	
	struct glrt_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structglrt__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structglrt__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structglrt__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structglrt__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pass2<doxid-structglrt__inform__type_iter_pass2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structglrt__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_regularized<doxid-structglrt__inform__type_obj_regularized>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structglrt__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`xpo_norm<doxid-structglrt__inform__type_xpo_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`leftmost<doxid-structglrt__inform__type_leftmost>`;
		bool :ref:`negative_curvature<doxid-structglrt__inform__type_negative_curvature>`;
		bool :ref:`hard_case<doxid-structglrt__inform__type_hard_case>`;
	};
.. _details-structglrt__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structglrt__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See :ref:`glrt_solve_problem <doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structglrt__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structglrt__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structglrt__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structglrt__inform__type_iter_pass2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; obj
.. _doxid-structglrt__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structglrt__inform__type_obj_regularized:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; multiplier
.. _doxid-structglrt__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the multiplier, $\sigma \|x\|^{p-2}$

.. index:: pair: variable; xpo_norm
.. _doxid-structglrt__inform__type_xpo_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` xpo_norm

the value of the norm $\|x\|_M$

.. index:: pair: variable; leftmost
.. _doxid-structglrt__inform__type_leftmost:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` leftmost

an estimate of the leftmost generalized eigenvalue of the pencil $(H,M)$

.. index:: pair: variable; negative_curvature
.. _doxid-structglrt__inform__type_negative_curvature:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool negative_curvature

was negative curvature encountered ?

.. index:: pair: variable; hard_case
.. _doxid-structglrt__inform__type_hard_case:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

did the hard case occur ?

