.. index:: pair: struct; ugo_inform_type
.. _doxid-structugo__inform__type:

ugo_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ugo.h>
	
	struct ugo_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structugo__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`eval_status<doxid-structugo__inform__type_eval_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structugo__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structugo__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structugo__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structugo__inform__type_f_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structugo__inform__type_g_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structugo__inform__type_h_eval>`;
		struct :ref:`ugo_time_type<doxid-structugo__time__type>` :ref:`time<doxid-structugo__inform__type_time>`;
	};
.. _details-structugo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structugo__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See UGO_solve for details

.. index:: pair: variable; eval_status
.. _doxid-structugo__inform__type_eval_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_status

evaluation status for reverse communication interface

.. index:: pair: variable; alloc_status
.. _doxid-structugo__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structugo__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structugo__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structugo__inform__type_f_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structugo__inform__type_g_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structugo__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; time
.. _doxid-structugo__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_time_type<doxid-structugo__time__type>` time

timings (see above)

