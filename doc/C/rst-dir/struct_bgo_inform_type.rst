.. index:: pair: struct; bgo_inform_type
.. _doxid-structbgo__inform__type:

bgo_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bgo.h>
	
	struct bgo_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structbgo__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structbgo__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structbgo__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structbgo__inform__type_f_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structbgo__inform__type_g_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structbgo__inform__type_h_eval>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structbgo__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structbgo__inform__type_norm_pg>`;
		struct :ref:`bgo_time_type<doxid-structbgo__time__type>` :ref:`time<doxid-structbgo__inform__type_time>`;
		struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` :ref:`ugo_inform<doxid-structbgo__inform__type_ugo_inform>`;
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>` :ref:`lhs_inform<doxid-structbgo__inform__type_lhs_inform>`;
		struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` :ref:`trb_inform<doxid-structbgo__inform__type_trb_inform>`;
	};
.. _details-structbgo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbgo__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See BGO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structbgo__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbgo__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; f_eval
.. _doxid-structbgo__inform__type_f_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structbgo__inform__type_g_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structbgo__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; obj
.. _doxid-structbgo__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; norm_pg
.. _doxid-structbgo__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; time
.. _doxid-structbgo__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bgo_time_type<doxid-structbgo__time__type>` time

timings (see above)

.. index:: pair: variable; ugo_inform
.. _doxid-structbgo__inform__type_ugo_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` ugo_inform

inform parameters for UGO

.. index:: pair: variable; lhs_inform
.. _doxid-structbgo__inform__type_lhs_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>` lhs_inform

inform parameters for LHS

.. index:: pair: variable; trb_inform
.. _doxid-structbgo__inform__type_trb_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` trb_inform

inform parameters for TRB

