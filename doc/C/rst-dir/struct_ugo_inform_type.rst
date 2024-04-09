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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structugo__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`eval_status<doxid-structugo__inform__type_1a7b88f94f39913a54e69ea23c12d3e4bd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structugo__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structugo__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structugo__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structugo__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structugo__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structugo__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a>`;
		struct :ref:`ugo_time_type<doxid-structugo__time__type>` :ref:`time<doxid-structugo__inform__type_1a9113f4cf33f961d4a1a97455a054eac6>`;
	};
.. _details-structugo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structugo__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See UGO_solve for details

.. index:: pair: variable; eval_status
.. _doxid-structugo__inform__type_1a7b88f94f39913a54e69ea23c12d3e4bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_status

evaluation status for reverse communication interface

.. index:: pair: variable; alloc_status
.. _doxid-structugo__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structugo__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structugo__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structugo__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structugo__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structugo__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; time
.. _doxid-structugo__inform__type_1a9113f4cf33f961d4a1a97455a054eac6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_time_type<doxid-structugo__time__type>` time

timings (see above)

