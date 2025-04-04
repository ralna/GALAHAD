.. index:: pair: struct; nodend_time_type
.. _doxid-structnodend__time__type:

nodend_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	struct nodend_time_type{T}
          total::T
          metis::T
          clock_total::T
          clock_metis::T

.. _details-structnodend__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structnodend__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; metis
.. _doxid-structnodend__time__type_metis:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T metis

the CPU time spent in the METIS package

.. index:: pair: variable; clock_total
.. _doxid-structnodend__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_metis
.. _doxid-structnodend__time__type_clock_metis:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_metis

the clock time spent in the METIS package
