.. index:: pair: table; dps_time_type
.. _doxid-structdps__time__type:

dps_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block
	
        struct dps_time_type{T}
          total::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structdps__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structdps__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total CPU time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structdps__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

CPU time spent reordering H prior to factorization.

.. index:: pair: variable; factorize
.. _doxid-structdps__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

CPU time spent factorizing H.

.. index:: pair: variable; solve
.. _doxid-structdps__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

CPU time spent solving the diagonal model system.

.. index:: pair: variable; clock_total
.. _doxid-structdps__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structdps__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time spent reordering H prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structdps__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing H

.. index:: pair: variable; clock_solve
.. _doxid-structdps__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time spent solving the diagonal model system

