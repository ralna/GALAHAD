.. index:: pair: table; rqs_history_type
.. _doxid-structrqs__history__type:

rqs_history_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_rqs.h>
	
	struct rqs_history_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lambda<doxid-structrqs__history__type_lambda>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structrqs__history__type_x_norm>`;
	};
.. _details-structrqs__history__type:

detailed documentation
----------------------

history derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structrqs__history__type_lambda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structrqs__history__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the corresponding value of $\|x(\lambda)\|_M$

