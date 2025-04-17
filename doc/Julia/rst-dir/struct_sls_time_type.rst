.. index:: pair: table; sls_time_type
.. _doxid-structsls__time__type:

sls_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sls_time_type{T}
          total::T
          analyse::T
          factorize::T
          solve::T
          order_external::T
          analyse_external::T
          factorize_external::T
          solve_external::T
          clock_total::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T
          clock_order_external::T
          clock_analyse_external::T
          clock_factorize_external::T
          clock_solve_external::T

.. _details-structsls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structsls__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the total cpu time spent in the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structsls__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the total cpu time spent in the factorization phase

.. index:: pair: variable; solve
.. _doxid-structsls__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the total cpu time spent in the solve phases

.. index:: pair: variable; order_external
.. _doxid-structsls__time__type_order_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T order_external

the total cpu time spent by the external solver in the ordering phase

.. index:: pair: variable; analyse_external
.. _doxid-structsls__time__type_analyse_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse_external

the total cpu time spent by the external solver in the analysis phase

.. index:: pair: variable; factorize_external
.. _doxid-structsls__time__type_factorize_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize_external

the total cpu time spent by the external solver in the factorization pha

.. index:: pair: variable; solve_external
.. _doxid-structsls__time__type_solve_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve_external

the total cpu time spent by the external solver in the solve phases

.. index:: pair: variable; clock_total
.. _doxid-structsls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structsls__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the total clock time spent in the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structsls__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the total clock time spent in the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structsls__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the total clock time spent in the solve phases

.. index:: pair: variable; clock_order_external
.. _doxid-structsls__time__type_clock_order_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_order_external

the total clock time spent by the external solver in the ordering phase

.. index:: pair: variable; clock_analyse_external
.. _doxid-structsls__time__type_clock_analyse_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse_external

the total clock time spent by the external solver in the analysis phase

.. index:: pair: variable; clock_factorize_external
.. _doxid-structsls__time__type_clock_factorize_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize_external

the total clock time spent by the external solver in the factorization p

.. index:: pair: variable; clock_solve_external
.. _doxid-structsls__time__type_clock_solve_external:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve_external

the total clock time spent by the external solver in the solve phases

