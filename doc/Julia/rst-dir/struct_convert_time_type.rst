.. index:: pair: table; convert_time_type
.. _doxid-structconvert__time__type:

convert_time_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct convert_time_type{T}
          total::T
          clock_total::T

.. _details-structconvert__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structconvert__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total cpu time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structconvert__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

