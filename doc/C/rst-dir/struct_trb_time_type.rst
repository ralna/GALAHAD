.. index:: pair: struct; trb_time_type
.. _doxid-structtrb__time__type:

trb_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trb.h>
	
	struct trb_time_type {
		// components
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structtrb__time__type_total>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`preprocess<doxid-structtrb__time__type_preprocess>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structtrb__time__type_analyse>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structtrb__time__type_factorize>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structtrb__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structtrb__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_preprocess<doxid-structtrb__time__type_clock_preprocess>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structtrb__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structtrb__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structtrb__time__type_clock_solve>`;
	};
.. _details-structtrb__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structtrb__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structtrb__time__type_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structtrb__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structtrb__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structtrb__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structtrb__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structtrb__time__type_clock_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structtrb__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structtrb__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structtrb__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing the search direction

