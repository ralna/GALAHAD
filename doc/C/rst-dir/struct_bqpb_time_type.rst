.. index:: pair: table; bqpb_time_type
.. _doxid-structbqpb__time__type:

bqpb_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	bqpb_time_type = {
		-- fields
	
		:ref:`total<doxid-structbqpb__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`,
		:ref:`preprocess<doxid-structbqpb__time__type_1a811a9183909ac7697f36d0ea8987715c>`,
		:ref:`find_dependent<doxid-structbqpb__time__type_1a30beab370e7d02ef42fe7ce99c55b147>`,
		:ref:`analyse<doxid-structbqpb__time__type_1a9c5b9155e1665977103d8c32881d9f00>`,
		:ref:`factorize<doxid-structbqpb__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`,
		:ref:`solve<doxid-structbqpb__time__type_1a4c971b10c915041b89daa05a29125376>`,
		:ref:`clock_total<doxid-structbqpb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`,
		:ref:`clock_preprocess<doxid-structbqpb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e>`,
		:ref:`clock_find_dependent<doxid-structbqpb__time__type_1a61d58c11f3f43b2171b6fc679e9845fa>`,
		:ref:`clock_analyse<doxid-structbqpb__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`,
		:ref:`clock_factorize<doxid-structbqpb__time__type_1ad3f0f50628260b90d6cf974e02f86192>`,
		:ref:`clock_solve<doxid-structbqpb__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`,
	}

.. _details-structbqpb__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbqpb__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	total

the total CPU time spent in the package

.. index:: pair: variable; preprocess
.. _doxid-structbqpb__time__type_1a811a9183909ac7697f36d0ea8987715c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	preprocess

the CPU time spent preprocessing the problem

.. index:: pair: variable; find_dependent
.. _doxid-structbqpb__time__type_1a30beab370e7d02ef42fe7ce99c55b147:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	find_dependent

the CPU time spent detecting linear dependencies

.. index:: pair: variable; analyse
.. _doxid-structbqpb__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structbqpb__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structbqpb__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	solve

the CPU time spent computing the search direction

.. index:: pair: variable; clock_total
.. _doxid-structbqpb__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_preprocess
.. _doxid-structbqpb__time__type_1a0c3b390c67037ef2fe8b4cf29e079e4e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_preprocess

the clock time spent preprocessing the problem

.. index:: pair: variable; clock_find_dependent
.. _doxid-structbqpb__time__type_1a61d58c11f3f43b2171b6fc679e9845fa:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_find_dependent

the clock time spent detecting linear dependencies

.. index:: pair: variable; clock_analyse
.. _doxid-structbqpb__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structbqpb__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structbqpb__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_solve

the clock time spent computing the search direction

