.. index:: pair: table; llst_history_type
.. _doxid-structllst__history__type:

llst_history_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_llst.h>
	
	struct llst_history_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lambda<doxid-structllst__history__type_lambda>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structllst__history__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structllst__history__type_r_norm>`;
	};
.. _details-structllst__history__type:

detailed documentation
----------------------

history derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structllst__history__type_lambda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structllst__history__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the corresponding value of $\|x(\lambda)\|_S$

.. index:: pair: variable; r_norm
.. _doxid-structllst__history__type_r_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

the corresponding value of $\|A x(\lambda) - b\|_2$

