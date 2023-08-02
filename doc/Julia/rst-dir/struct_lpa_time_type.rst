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
.. _doxid-structlpa__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structlpa__time__type_1a811a9183909ac7697f36d0ea8987715c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; clock_total
.. _doxid-structlpa__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structlpa__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

