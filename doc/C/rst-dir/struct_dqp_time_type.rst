.. index:: pair: table; dqp_time_type
.. _doxid-structdqp__time__type:

dqp_time_type structure
-----------------------

.. toctree::
	:hidden:

time derived type as a C struct :ref:`More...<details-structdqp__time__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	dqp_time_type = {
		-- fields
	
		:ref:`total<doxid-structdqp__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`,
		:ref:`preprocess<doxid-structdqp__time__type_1a811a9183909ac7697f36d0ea8987715c>`,
		:ref:`find_dependent<doxid-structdqp__time__type_1a30beab370e7d02ef42fe7ce99c55b147>`,
		:ref:`analyse<doxid-structdqp__time__type_1a9c5b9155e1665977103d8c32881d9f00>`,
		:ref:`factorize<doxid-structdqp__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`,
		:ref:`solve<doxid-structdqp__time__type_1a4c971b10c915041b89daa05a29125376>`,
		:ref:`search<doxid-structdqp__time__type_1a0fcf9ee6a80e15c385eb3488553648f8>`,
		:ref:`clock_total<doxid-structdqp__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`,
		:ref:`clock_preprocess<doxid-structdqp__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e>`,
		:ref:`clock_find_dependent<doxid-structdqp__time__type_1a61d58c11f3f43b2171b6fc679e9845fa>`,
		:ref:`clock_analyse<doxid-structdqp__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`,
		:ref:`clock_factorize<doxid-structdqp__time__type_1ad3f0f50628260b90d6cf974e02f86192>`,
		:ref:`clock_solve<doxid-structdqp__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`,
		:ref:`clock_search<doxid-structdqp__time__type_1a82c4232bdc5b5cdfdc982b380c7a6e9c>`,
	}

.. _details-structdqp__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structdqp__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structdqp__time__type_1a811a9183909ac7697f36d0ea8987715c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structdqp__time__type_1a30beab370e7d02ef42fe7ce99c55b147:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structdqp__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structdqp__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structdqp__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	solve

the CPU time spent computing the search direction

.. index:: pair: variable; search
.. _doxid-structdqp__time__type_1a0fcf9ee6a80e15c385eb3488553648f8:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	search

the CPU time spent in the linesearch

.. index:: pair: variable; clock_total
.. _doxid-structdqp__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structdqp__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structdqp__time__type_1a61d58c11f3f43b2171b6fc679e9845fa:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structdqp__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structdqp__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structdqp__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_solve

the clock time spent computing the search direction

.. index:: pair: variable; clock_search
.. _doxid-structdqp__time__type_1a82c4232bdc5b5cdfdc982b380c7a6e9c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_search

the clock time spent in the linesearch

