.. index:: pair: table; llsr_history_type
.. _doxid-structllsr__history__type:

llsr_history_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct llsr_history_type{T}
          lambda::T
          x_norm::T
          r_norm::T
	
.. _details-structllsr__history__type:

detailed documentation
----------------------

history derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structllsr__history__type_lambda:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structllsr__history__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the corresponding value of $\|x(\lambda)\|_M$

.. index:: pair: variable; r_norm
.. _doxid-structllsr__history__type_r_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

the corresponding value of $\|A x(\lambda) - b\|_2$

