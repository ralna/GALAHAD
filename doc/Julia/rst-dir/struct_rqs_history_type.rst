.. index:: pair: table; rqs_history_type
.. _doxid-structrqs__history__type:

rqs_history_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct rqs_history_type{T}
          lambda::T
          x_norm::T

.. _details-structrqs__history__type:

detailed documentation
----------------------

history derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structrqs__history__type_1a69856cb11373bfb6f36d8a28df6dd08f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structrqs__history__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the corresponding value of $\|x(\lambda)\|_M$

