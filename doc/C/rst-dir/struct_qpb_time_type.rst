.. index:: pair: struct; qpb_time_type
.. _doxid-structqpb__time__type:

qpb_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_qpb.h>
	
	struct qpb_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structqpb__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`preprocess<doxid-structqpb__time__type_preprocess>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`find_dependent<doxid-structqpb__time__type_find_dependent>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structqpb__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structqpb__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structqpb__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_total<doxid-structqpb__time__type_phase1_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_analyse<doxid-structqpb__time__type_phase1_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_factorize<doxid-structqpb__time__type_phase1_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_solve<doxid-structqpb__time__type_phase1_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structqpb__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_preprocess<doxid-structqpb__time__type_clock_preprocess>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_find_dependent<doxid-structqpb__time__type_clock_find_dependent>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structqpb__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structqpb__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structqpb__time__type_clock_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_total<doxid-structqpb__time__type_clock_phase1_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_analyse<doxid-structqpb__time__type_clock_phase1_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_factorize<doxid-structqpb__time__type_clock_phase1_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_solve<doxid-structqpb__time__type_clock_phase1_solve>`;
	};
.. _details-structqpb__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structqpb__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structqpb__time__type_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structqpb__time__type_find_dependent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structqpb__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorizatio

.. index:: pair: variable; factorize
.. _doxid-structqpb__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structqpb__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

the CPU time spent computing the search direction

.. index:: pair: variable; phase1_total
.. _doxid-structqpb__time__type_phase1_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_total

the total CPU time spent in the initial-point phase of the package

.. index:: pair: variable; phase1_analyse
.. _doxid-structqpb__time__type_phase1_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_analyse

the CPU time spent analysing the required matrices prior to factorizatio in the inital-point phase

.. index:: pair: variable; phase1_factorize
.. _doxid-structqpb__time__type_phase1_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_factorize

the CPU time spent factorizing the required matrices in the inital-point phase

.. index:: pair: variable; phase1_solve
.. _doxid-structqpb__time__type_phase1_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_solve

the CPU time spent computing the search direction in the inital-point ph

.. index:: pair: variable; clock_total
.. _doxid-structqpb__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structqpb__time__type_clock_preprocess:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structqpb__time__type_clock_find_dependent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structqpb__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structqpb__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structqpb__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing the search direction

.. index:: pair: variable; clock_phase1_total
.. _doxid-structqpb__time__type_clock_phase1_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_total

the total clock time spent in the initial-point phase of the package

.. index:: pair: variable; clock_phase1_analyse
.. _doxid-structqpb__time__type_clock_phase1_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_analyse

the clock time spent analysing the required matrices prior to factorizat in the inital-point phase

.. index:: pair: variable; clock_phase1_factorize
.. _doxid-structqpb__time__type_clock_phase1_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_factorize

the clock time spent factorizing the required matrices in the inital-poi phase

.. index:: pair: variable; clock_phase1_solve
.. _doxid-structqpb__time__type_clock_phase1_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_solve

the clock time spent computing the search direction in the inital-point

