.. index:: pair: struct; qpa_time_type
.. _doxid-structqpa__time__type:

qpa_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpa_time_type{T}
          total::T
          preprocess::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_preprocess::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structqpa__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structqpa__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structqpa__time__type_1a811a9183909ac7697f36d0ea8987715c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structqpa__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the CPU time spent analysing the required matrices prior to factorizatio

.. index:: pair: variable; factorize
.. _doxid-structqpa__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structqpa__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structqpa__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structqpa__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structqpa__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structqpa__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structqpa__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent computing the search direction

