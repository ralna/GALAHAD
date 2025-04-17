.. index:: pair: struct; ugo_time_type
.. _doxid-structugo__time__type:

ugo_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ugo_time_type{T}
          total::Float32
          clock_total::T

.. _details-structugo__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structugo__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

the total CPU time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structugo__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package
