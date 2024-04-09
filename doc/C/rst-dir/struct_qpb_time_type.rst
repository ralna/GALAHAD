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
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structqpb__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`preprocess<doxid-structqpb__time__type_1a811a9183909ac7697f36d0ea8987715c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`find_dependent<doxid-structqpb__time__type_1a30beab370e7d02ef42fe7ce99c55b147>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structqpb__time__type_1a9c5b9155e1665977103d8c32881d9f00>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structqpb__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structqpb__time__type_1a4c971b10c915041b89daa05a29125376>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_total<doxid-structqpb__time__type_1a95dcb61a016bdfc124262e5fd9f20096>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_analyse<doxid-structqpb__time__type_1a7952377524e023b6b0c848e34ea6f9da>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_factorize<doxid-structqpb__time__type_1af1eaf09b990768d50a83a025db485609>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`phase1_solve<doxid-structqpb__time__type_1ab446d6f2a44a1ae0e397383cf5dabebe>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structqpb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_preprocess<doxid-structqpb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_find_dependent<doxid-structqpb__time__type_1a61d58c11f3f43b2171b6fc679e9845fa>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structqpb__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structqpb__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structqpb__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_total<doxid-structqpb__time__type_1a4a099c3231600eb61ee7b5ace493b67e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_analyse<doxid-structqpb__time__type_1ae30bf39457aaaef3ceecc84a38a36243>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_factorize<doxid-structqpb__time__type_1a95f8dbbf28776a5cff044ad7145d0c99>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_phase1_solve<doxid-structqpb__time__type_1a2cdc17b98a4e264d7790cf645358946e>`;
	};
.. _details-structqpb__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structqpb__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structqpb__time__type_1a811a9183909ac7697f36d0ea8987715c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structqpb__time__type_1a30beab370e7d02ef42fe7ce99c55b147:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structqpb__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorizatio

.. index:: pair: variable; factorize
.. _doxid-structqpb__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structqpb__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

the CPU time spent computing the search direction

.. index:: pair: variable; phase1_total
.. _doxid-structqpb__time__type_1a95dcb61a016bdfc124262e5fd9f20096:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_total

the total CPU time spent in the initial-point phase of the package

.. index:: pair: variable; phase1_analyse
.. _doxid-structqpb__time__type_1a7952377524e023b6b0c848e34ea6f9da:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_analyse

the CPU time spent analysing the required matrices prior to factorizatio in the inital-point phase

.. index:: pair: variable; phase1_factorize
.. _doxid-structqpb__time__type_1af1eaf09b990768d50a83a025db485609:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_factorize

the CPU time spent factorizing the required matrices in the inital-point phase

.. index:: pair: variable; phase1_solve
.. _doxid-structqpb__time__type_1ab446d6f2a44a1ae0e397383cf5dabebe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` phase1_solve

the CPU time spent computing the search direction in the inital-point ph

.. index:: pair: variable; clock_total
.. _doxid-structqpb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structqpb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structqpb__time__type_1a61d58c11f3f43b2171b6fc679e9845fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structqpb__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structqpb__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structqpb__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing the search direction

.. index:: pair: variable; clock_phase1_total
.. _doxid-structqpb__time__type_1a4a099c3231600eb61ee7b5ace493b67e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_total

the total clock time spent in the initial-point phase of the package

.. index:: pair: variable; clock_phase1_analyse
.. _doxid-structqpb__time__type_1ae30bf39457aaaef3ceecc84a38a36243:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_analyse

the clock time spent analysing the required matrices prior to factorizat in the inital-point phase

.. index:: pair: variable; clock_phase1_factorize
.. _doxid-structqpb__time__type_1a95f8dbbf28776a5cff044ad7145d0c99:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_factorize

the clock time spent factorizing the required matrices in the inital-poi phase

.. index:: pair: variable; clock_phase1_solve
.. _doxid-structqpb__time__type_1a2cdc17b98a4e264d7790cf645358946e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_phase1_solve

the clock time spent computing the search direction in the inital-point

