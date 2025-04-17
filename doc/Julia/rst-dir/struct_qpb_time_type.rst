.. index:: pair: struct; qpb_time_type
.. _doxid-structqpb__time__type:

qpb_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpb_time_type{T}
          total::T
          preprocess::T
          find_dependent::T
          analyse::T
          factorize::T
          solve::T
          phase1_total::T
          phase1_analyse::T
          phase1_factorize::T
          phase1_solve::T
          clock_total::T
          clock_preprocess::T
          clock_find_dependent::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T
          clock_phase1_total::T
          clock_phase1_analyse::T
          clock_phase1_factorize::T
          clock_phase1_solve::T

.. _details-structqpb__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structqpb__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structqpb__time__type_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structqpb__time__type_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structqpb__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the CPU time spent analysing the required matrices prior to factorizatio

.. index:: pair: variable; factorize
.. _doxid-structqpb__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structqpb__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent computing the search direction

.. index:: pair: variable; phase1_total
.. _doxid-structqpb__time__type_phase1_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T phase1_total

the total CPU time spent in the initial-point phase of the package

.. index:: pair: variable; phase1_analyse
.. _doxid-structqpb__time__type_phase1_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T phase1_analyse

the CPU time spent analysing the required matrices prior to factorizatio in the inital-point phase

.. index:: pair: variable; phase1_factorize
.. _doxid-structqpb__time__type_phase1_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T phase1_factorize

the CPU time spent factorizing the required matrices in the inital-point phase

.. index:: pair: variable; phase1_solve
.. _doxid-structqpb__time__type_phase1_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T phase1_solve

the CPU time spent computing the search direction in the inital-point ph

.. index:: pair: variable; clock_total
.. _doxid-structqpb__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structqpb__time__type_clock_preprocess:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structqpb__time__type_clock_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structqpb__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structqpb__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structqpb__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

.. index:: pair: variable; clock_phase1_total
.. _doxid-structqpb__time__type_clock_phase1_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_phase1_total

the total clock time spent in the initial-point phase of the package

.. index:: pair: variable; clock_phase1_analyse
.. _doxid-structqpb__time__type_clock_phase1_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_phase1_analyse

the clock time spent analysing the required matrices prior to factorizat in the inital-point phase

.. index:: pair: variable; clock_phase1_factorize
.. _doxid-structqpb__time__type_clock_phase1_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_phase1_factorize

the clock time spent factorizing the required matrices in the inital-poi phase

.. index:: pair: variable; clock_phase1_solve
.. _doxid-structqpb__time__type_clock_phase1_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_phase1_solve

the clock time spent computing the search direction in the inital-point

