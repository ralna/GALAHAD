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
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structtrb__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`preprocess<doxid-structtrb__time__type_1adc5369028902776a12fe8d393be67174>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structtrb__time__type_1a0ca2b20748c7749a77d684124011c531>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structtrb__time__type_1ab7eecce4b013c87e490b8984c74c59c3>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structtrb__time__type_1a6356532c25755a6e5fedee1a7d703949>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structtrb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_preprocess<doxid-structtrb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structtrb__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structtrb__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structtrb__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`;
	};
.. _details-structtrb__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structtrb__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structtrb__time__type_1adc5369028902776a12fe8d393be67174:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structtrb__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structtrb__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structtrb__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structtrb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structtrb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structtrb__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structtrb__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structtrb__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing the search direction

