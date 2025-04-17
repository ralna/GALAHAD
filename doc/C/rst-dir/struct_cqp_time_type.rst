.. index:: pair: struct; cqp_time_type
.. _doxid-structcqp__time__type:

cqp_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_cqp.h>
	
	struct cqp_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structcqp__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`preprocess<doxid-structcqp__time__type_preprocess>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`find_dependent<doxid-structcqp__time__type_find_dependent>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structcqp__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structcqp__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structcqp__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structcqp__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_preprocess<doxid-structcqp__time__type_clock_preprocess>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_find_dependent<doxid-structcqp__time__type_clock_find_dependent>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structcqp__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structcqp__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structcqp__time__type_clock_solve>`;
	};
.. _details-structcqp__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structcqp__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structcqp__time__type_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structcqp__time__type_find_dependent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structcqp__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structcqp__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structcqp__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structcqp__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structcqp__time__type_clock_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structcqp__time__type_clock_find_dependent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structcqp__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structcqp__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structcqp__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing the search direction

