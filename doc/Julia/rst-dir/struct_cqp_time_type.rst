.. index:: pair: struct; cqp_time_type
.. _doxid-structcqp__time__type:

cqp_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	struct cqp_time_type{T}
          total::T
          preprocess::T
          find_dependent::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_preprocess::T
          clock_find_dependent::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structcqp__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structcqp__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structcqp__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structcqp__time__type_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structcqp__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structcqp__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structcqp__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structcqp__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structcqp__time__type_clock_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structcqp__time__type_clock_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structcqp__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structcqp__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structcqp__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

