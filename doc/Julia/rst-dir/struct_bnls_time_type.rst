.. index:: pair: struct; bnls_time_type
.. _doxid-structbnls__time__type:

bnls_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bnls_time_type{T}
          total::T
          blls::T
          bllsb::T
          clock_total::T
          clock_blls::T
          clock_bllsb::T

.. _details-structbnls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbnls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; blls
.. _doxid-structbnls__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T blls

the CPU time spent in the ``bllsb`` package

.. index:: pair: variable; bllsb
.. _doxid-structbnls__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T bllsb

the CPU time spent in the ``bllsb`` package

.. index:: pair: variable; clock_total
.. _doxid-structbnls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_blls
.. _doxid-structbnls__time__type_clock_blls:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_blls

the clock time spent in the ``blls`` package

.. index:: pair: variable; clock_bllsb
.. _doxid-structbnls__time__type_clock_bllsb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_bllsb

the clock time spent in the ``bllsb`` package


