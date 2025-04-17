.. index:: pair: struct; bgo_time_type
.. _doxid-structbgo__time__type:

bgo_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bgo_time_type{T}
          total::Float32
          univariate_global::Float32
          multivariate_local::Float32
          clock_total::T
          clock_univariate_global::T
          clock_multivariate_local::T

.. _details-structbgo__time__type:

detailed documentation
----------------------

time derived type as a Julia structure


components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbgo__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

the total CPU time spent in the package

.. index:: pair: variable; univariate_global
.. _doxid-structbgo__time__type_univariate_global:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 univariate_global

the CPU time spent performing univariate global optimization

.. index:: pair: variable; multivariate_local
.. _doxid-structbgo__time__type_multivariate_local:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 multivariate_local

the CPU time spent performing multivariate local optimization

.. index:: pair: variable; clock_total
.. _doxid-structbgo__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_univariate_global
.. _doxid-structbgo__time__type_clock_univariate_global:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_univariate_global

the clock time spent performing univariate global optimization

.. index:: pair: variable; clock_multivariate_local
.. _doxid-structbgo__time__type_clock_multivariate_local:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_multivariate_local

the clock time spent performing multivariate local optimization

