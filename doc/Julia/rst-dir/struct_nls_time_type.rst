.. index:: pair: struct; nls_time_type
.. _doxid-structnls__time__type:

nls_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nls_time_type{T}
          total::Float32
          preprocess::Float32
          analyse::Float32
          factorize::Float32
          solve::Float32
          clock_total::T
          clock_preprocess::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structnls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structnls__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structnls__time__type_1adc5369028902776a12fe8d393be67174:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structnls__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structnls__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structnls__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structnls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structnls__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structnls__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structnls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structnls__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

