.. index:: pair: struct; dgo_inform_type
.. _doxid-structdgo__inform__type:

dgo_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dgo.h>
	
	struct dgo_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structdgo__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structdgo__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structdgo__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structdgo__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structdgo__inform__type_f_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structdgo__inform__type_g_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structdgo__inform__type_h_eval>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structdgo__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structdgo__inform__type_norm_pg>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`length_ratio<doxid-structdgo__inform__type_length_ratio>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`f_gap<doxid-structdgo__inform__type_f_gap>`;
		char :ref:`why_stop<doxid-structdgo__inform__type_why_stop>`[2];
		struct :ref:`dgo_time_type<doxid-structdgo__time__type>` :ref:`time<doxid-structdgo__inform__type_time>`;
		struct :ref:`hash_inform_type<doxid-structhash__inform__type>` :ref:`hash_inform<doxid-structdgo__inform__type_hash_inform>`;
		struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` :ref:`ugo_inform<doxid-structdgo__inform__type_ugo_inform>`;
		struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` :ref:`trb_inform<doxid-structdgo__inform__type_trb_inform>`;
	};
.. _details-structdgo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct


components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdgo__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See DGO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdgo__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structdgo__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structdgo__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structdgo__inform__type_f_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structdgo__inform__type_g_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structdgo__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; obj
.. _doxid-structdgo__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by DGO_solve

.. index:: pair: variable; norm_pg
.. _doxid-structdgo__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by DGO_solve

.. index:: pair: variable; length_ratio
.. _doxid-structdgo__inform__type_length_ratio:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` length_ratio

the ratio of the final to the initial box lengths

.. index:: pair: variable; f_gap
.. _doxid-structdgo__inform__type_f_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` f_gap

the gap between the best objective value found and the lowest bound

.. index:: pair: variable; why_stop
.. _doxid-structdgo__inform__type_why_stop:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char why_stop[2]

why did the iteration stop? This wil be 'D' if the box length is small enough, 'F' if the objective gap is small enough, and ' ' otherwise

.. index:: pair: variable; time
.. _doxid-structdgo__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dgo_time_type<doxid-structdgo__time__type>` time

timings (see above)

.. index:: pair: variable; hash_inform
.. _doxid-structdgo__inform__type_hash_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`hash_inform_type<doxid-structhash__inform__type>` hash_inform

inform parameters for HASH

.. index:: pair: variable; ugo_inform
.. _doxid-structdgo__inform__type_ugo_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` ugo_inform

inform parameters for UGO

.. index:: pair: variable; trb_inform
.. _doxid-structdgo__inform__type_trb_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` trb_inform

inform parameters for TRB

