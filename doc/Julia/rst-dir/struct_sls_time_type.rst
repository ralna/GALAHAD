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
.. _doxid-structsls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

the total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structsls__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

the total cpu time spent in the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structsls__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

the total cpu time spent in the factorization phase

.. index:: pair: variable; solve
.. _doxid-structsls__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

the total cpu time spent in the solve phases

.. index:: pair: variable; order_external
.. _doxid-structsls__time__type_1ac9e0db16df3b373e6192360c84f71aab:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T order_external

the total cpu time spent by the external solver in the ordering phase

.. index:: pair: variable; analyse_external
.. _doxid-structsls__time__type_1a9b4f1f33374092b099ab7baad8d1d6ac:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse_external

the total cpu time spent by the external solver in the analysis phase

.. index:: pair: variable; factorize_external
.. _doxid-structsls__time__type_1a87f0b3565b139ff7baa76ce830a92964:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize_external

the total cpu time spent by the external solver in the factorization pha

.. index:: pair: variable; solve_external
.. _doxid-structsls__time__type_1a8cbd0409a559f5e24c15591cf8d53eeb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve_external

the total cpu time spent by the external solver in the solve phases

.. index:: pair: variable; clock_total
.. _doxid-structsls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structsls__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

the total clock time spent in the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structsls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

the total clock time spent in the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structsls__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

the total clock time spent in the solve phases

.. index:: pair: variable; clock_order_external
.. _doxid-structsls__time__type_1a7babcea658f1454261df6b8acc24be9b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_order_external

the total clock time spent by the external solver in the ordering phase

.. index:: pair: variable; clock_analyse_external
.. _doxid-structsls__time__type_1ac479ec45cbef59b5aff36cc55861dc63:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse_external

the total clock time spent by the external solver in the analysis phase

.. index:: pair: variable; clock_factorize_external
.. _doxid-structsls__time__type_1aab3765fc7d7de7a6148eb861ebd8fa31:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize_external

the total clock time spent by the external solver in the factorization p

.. index:: pair: variable; clock_solve_external
.. _doxid-structsls__time__type_1abcd8605d13ed5379a535b305c69ee1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve_external

the total clock time spent by the external solver in the solve phases

