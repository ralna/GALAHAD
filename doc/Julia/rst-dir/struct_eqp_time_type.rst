.. index:: pair: struct; eqp_time_type
.. _doxid-structeqp__time__type:

eqp_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct eqp_time_type{T}
          total::T
          find_dependent::T
          factorize::T
          solve::T
          solve_inter::T
          clock_total::T
          clock_find_dependent::T
          clock_factorize::T
          clock_solve::T
	
.. _details-structeqp__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structeqp__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; find_dependent
.. _doxid-structeqp__time__type_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; factorize
.. _doxid-structeqp__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structeqp__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent computing the search direction

.. index:: pair: variable; solve_inter
.. _doxid-structeqp__time__type_solve_inter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve_inter

see solve

.. index:: pair: variable; clock_total
.. _doxid-structeqp__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_find_dependent
.. _doxid-structeqp__time__type_clock_find_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_factorize
.. _doxid-structeqp__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structeqp__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

