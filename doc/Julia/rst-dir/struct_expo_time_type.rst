.. index:: pair: struct; expo_time_type
.. _doxid-structexpo__time__type:

expo_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct expo_time_type{T}
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

.. _details-structexpo__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structexpo__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structexpo__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structexpo__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structexpo__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structexpo__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structexpo__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structexpo__time__type_clock_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structexpo__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structexpo__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structexpo__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

