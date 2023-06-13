.. index:: pair: table; arc_time_type
.. _doxid-structarc__time__type:

arc_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	arc_time_type = {
		-- fields
	
		:ref:`total<doxid-structarc__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`,
		:ref:`preprocess<doxid-structarc__time__type_1adc5369028902776a12fe8d393be67174>`,
		:ref:`analyse<doxid-structarc__time__type_1a0ca2b20748c7749a77d684124011c531>`,
		:ref:`factorize<doxid-structarc__time__type_1ab7eecce4b013c87e490b8984c74c59c3>`,
		:ref:`solve<doxid-structarc__time__type_1a6356532c25755a6e5fedee1a7d703949>`,
		:ref:`clock_total<doxid-structarc__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`,
		:ref:`clock_preprocess<doxid-structarc__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e>`,
		:ref:`clock_analyse<doxid-structarc__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`,
		:ref:`clock_factorize<doxid-structarc__time__type_1ad3f0f50628260b90d6cf974e02f86192>`,
		:ref:`clock_solve<doxid-structarc__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`,
	}

.. _details-structarc__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structarc__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structarc__time__type_1adc5369028902776a12fe8d393be67174:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; analyse
.. _doxid-structarc__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	analyse

the CPU time spent analysing the required matrices prior to factorizatio

.. index:: pair: variable; factorize
.. _doxid-structarc__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structarc__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structarc__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structarc__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_analyse
.. _doxid-structarc__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structarc__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structarc__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_solve

the clock time spent computing the search direction

