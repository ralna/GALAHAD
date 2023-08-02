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
.. _doxid-structeqp__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; find_dependent
.. _doxid-structeqp__time__type_1a30beab370e7d02ef42fe7ce99c55b147:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; factorize
.. _doxid-structeqp__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structeqp__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent computing the search direction

.. index:: pair: variable; solve_inter
.. _doxid-structeqp__time__type_1a58c53b147b5bfa0b551a6e954455ee37:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve_inter

see solve

.. index:: pair: variable; clock_total
.. _doxid-structeqp__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_find_dependent
.. _doxid-structeqp__time__type_1a61d58c11f3f43b2171b6fc679e9845fa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_factorize
.. _doxid-structeqp__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structeqp__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

