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
	
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`lambda<doxid-structllst__history__type_1a69856cb11373bfb6f36d8a28df6dd08f>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`x_norm<doxid-structllst__history__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`r_norm<doxid-structllst__history__type_1ae908410fabf891cfd89626c3605c38ca>`;
	};
.. _details-structllst__history__type:

detailed documentation
----------------------

history derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structllst__history__type_1a69856cb11373bfb6f36d8a28df6dd08f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` lambda

the value of :math:`\lambda`

.. index:: pair: variable; x_norm
.. _doxid-structllst__history__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_norm

the corresponding value of :math:`\|x(\lambda)\|_S`

.. index:: pair: variable; r_norm
.. _doxid-structllst__history__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` r_norm

the corresponding value of :math:`\|A x(\lambda) - b\|_2`

