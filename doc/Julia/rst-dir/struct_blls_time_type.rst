.. index:: pair: struct; blls_time_type
.. _doxid-structblls__time__type:

blls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia

        struct blls_time_type{T}
          total::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structblls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structblls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total CPU time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structblls__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structblls__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structblls__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the CPU time spent in the linear solution phase

.. index:: pair: variable; clock_total
.. _doxid-structblls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structblls__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structblls__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structblls__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the clock time spent in the linear solution phase

