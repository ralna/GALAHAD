.. index:: pair: struct; blls_time_type
.. _doxid-structblls__time__type:

blls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia

        struct blls_time_type{T}
          total::T
          clock_total::T

.. _details-structblls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structblls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structblls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package
