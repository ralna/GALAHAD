.. index:: pair: struct; lpa_time_type
.. _doxid-structlpa__time__type:

lpa_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lpa_time_type{T}
          total::T
          preprocess::T
          clock_total::T
          clock_preprocess::T

.. _details-structlpa__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structlpa__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structlpa__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; clock_total
.. _doxid-structlpa__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structlpa__time__type_clock_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

