.. index:: pair: table; rqs_time_type
.. _doxid-structrqs__time__type:

rqs_time_type structure
-----------------------

.. toctree::
	:hidden:

.. _details-structrqs__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; rqs_time_type
.. _doxid-structrqs__time__type:

table rqs_time_type
===================


.. toctree::
	:hidden:

Overview
~~~~~~~~

time derived type as a C struct :ref:`More...<details-structrqs__time__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	rqs_time_type = {
		-- fields
	
		:ref:`total<doxid-structrqs__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`,
		:ref:`assemble<doxid-structrqs__time__type_1ae84d232eee798a974ebaeb9c82d623f4>`,
		:ref:`analyse<doxid-structrqs__time__type_1a9c5b9155e1665977103d8c32881d9f00>`,
		:ref:`factorize<doxid-structrqs__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`,
		:ref:`solve<doxid-structrqs__time__type_1a4c971b10c915041b89daa05a29125376>`,
		:ref:`clock_total<doxid-structrqs__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`,
		:ref:`clock_assemble<doxid-structrqs__time__type_1a4df2b92cea9269b8f8cad7024b83a10d>`,
		:ref:`clock_analyse<doxid-structrqs__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`,
		:ref:`clock_factorize<doxid-structrqs__time__type_1ad3f0f50628260b90d6cf974e02f86192>`,
		:ref:`clock_solve<doxid-structrqs__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`,
	}

.. _details-structrqs__time__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

time derived type as a C struct

Fields
------

.. index:: pair: variable; total
.. _doxid-structrqs__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	total

total CPU time spent in the package

.. index:: pair: variable; assemble
.. _doxid-structrqs__time__type_1ae84d232eee798a974ebaeb9c82d623f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	assemble

CPU time spent building :math:`H + \lambda M`.

.. index:: pair: variable; analyse
.. _doxid-structrqs__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	analyse

CPU time spent reordering :math:`H + \lambda M` prior to factorization.

.. index:: pair: variable; factorize
.. _doxid-structrqs__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorize

CPU time spent factorizing :math:`H + \lambda M`.

.. index:: pair: variable; solve
.. _doxid-structrqs__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	solve

CPU time spent solving linear systems inolving :math:`H + \lambda M`.

.. index:: pair: variable; clock_total
.. _doxid-structrqs__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structrqs__time__type_1a4df2b92cea9269b8f8cad7024b83a10d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_assemble

clock time spent building :math:`H + \lambda M`

.. index:: pair: variable; clock_analyse
.. _doxid-structrqs__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_analyse

clock time spent reordering :math:`H + \lambda M` prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structrqs__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_factorize

clock time spent factorizing :math:`H + \lambda M`

.. index:: pair: variable; clock_solve
.. _doxid-structrqs__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	clock_solve

clock time spent solving linear systems inolving :math:`H + \lambda M`

