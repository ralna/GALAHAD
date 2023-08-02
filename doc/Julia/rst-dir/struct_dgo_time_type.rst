.. index:: pair: struct; dgo_time_type
.. _doxid-structdgo__time__type:

dgo_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct dgo_time_type{T}
          total::Float32
          univariate_global::Float32
          multivariate_local::Float32
          clock_total::T
          clock_univariate_global::T
          clock_multivariate_local::T

.. _details-structdgo__time__type:

detailed documentation
----------------------

time derived type as a Julia structure


components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structdgo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

the total CPU time spent in the package

.. index:: pair: variable; univariate_global
.. _doxid-structdgo__time__type_1ae803cab9cf49e3b9f259415e254f7a8e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 univariate_global

the CPU time spent performing univariate global optimization

.. index:: pair: variable; multivariate_local
.. _doxid-structdgo__time__type_1ae3473e3e6e1f5482c642784f7e5b85e7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 multivariate_local

the CPU time spent performing multivariate local optimization

.. index:: pair: variable; clock_total
.. _doxid-structdgo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_univariate_global
.. _doxid-structdgo__time__type_1a35fea348c7aed26574dec4efbd9a7107:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_univariate_global

the clock time spent performing univariate global optimization

.. index:: pair: variable; clock_multivariate_local
.. _doxid-structdgo__time__type_1a7e6ac9410dc0d6af0a020612ad4fceb0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_multivariate_local

the clock time spent performing multivariate local optimization

