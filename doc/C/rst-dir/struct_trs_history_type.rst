.. index:: pair: table; trs_history_type
.. _doxid-structtrs__history__type:

trs_history_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trs.h>
	
	struct trs_history_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lambda<doxid-structtrs__history__type_lambda>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structtrs__history__type_x_norm>`;
	};
.. _details-structtrs__history__type:

detailed documentation
----------------------

history derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structtrs__history__type_lambda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structtrs__history__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the corresponding value of $\|x(\lambda)\|_M$

