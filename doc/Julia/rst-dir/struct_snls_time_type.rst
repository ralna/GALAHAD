.. index:: pair: struct; snls_time_type
.. _doxid-structsnls__time__type:

snls_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct snls_time_type{T}
          total::T
          slls::T
          sllsb::T
          clock_total::T
          clock_slls::T
          clock_sllsb::T

.. _details-structsnls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsnls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; slls
.. _doxid-structsnls__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T slls

the CPU time spent in the ``sllsb`` package

.. index:: pair: variable; sllsb
.. _doxid-structsnls__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T sllsb

the CPU time spent in the ``sllsb`` package

.. index:: pair: variable; clock_total
.. _doxid-structsnls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_slls
.. _doxid-structsnls__time__type_clock_slls:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_slls

the clock time spent in the ``slls`` package

.. index:: pair: variable; clock_sllsb
.. _doxid-structsnls__time__type_clock_sllsb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_sllsb

the clock time spent in the ``sllsb`` package


