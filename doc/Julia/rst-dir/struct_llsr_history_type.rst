.. index:: pair: table; llsr_history_type
.. _doxid-structllsr__history__type:

llsr_history_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	#include <galahad_llsr.h>
	
	struct llsr_history_type {
		// fields
	
		T :ref:`lambda<doxid-structllsr__history__type_1a69856cb11373bfb6f36d8a28df6dd08f>`;
		T :ref:`x_norm<doxid-structllsr__history__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`;
		T :ref:`r_norm<doxid-structllsr__history__type_1ae908410fabf891cfd89626c3605c38ca>`;
	};
.. _details-structllsr__history__type:

detailed documentation
----------------------

history derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structllsr__history__type_1a69856cb11373bfb6f36d8a28df6dd08f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lambda

the value of :math:`\lambda`

.. index:: pair: variable; x_norm
.. _doxid-structllsr__history__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the corresponding value of :math:`\|x(\lambda)\|_M`

.. index:: pair: variable; r_norm
.. _doxid-structllsr__history__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

the corresponding value of :math:`\|A x(\lambda) - b\|_2`

