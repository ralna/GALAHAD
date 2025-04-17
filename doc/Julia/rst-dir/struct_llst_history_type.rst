.. index:: pair: table; llst_history_type
.. _doxid-structllst__history__type:

llst_history_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct llst_history_type{T}
          lambda::T
          x_norm::T
          r_norm::T

.. _details-structllst__history__type:

detailed documentation
----------------------

history derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; lambda
.. _doxid-structllst__history__type_lambda:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lambda

the value of $\lambda$

.. index:: pair: variable; x_norm
.. _doxid-structllst__history__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the corresponding value of $\|x(\lambda)\|_S$

.. index:: pair: variable; r_norm
.. _doxid-structllst__history__type_r_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

the corresponding value of $\|A x(\lambda) - b\|_2$

