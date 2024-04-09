.. index:: pair: table; rqs_time_type
.. _doxid-structrqs__time__type:

rqs_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_rqs.h>
	
	struct rqs_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structrqs__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`assemble<doxid-structrqs__time__type_1ae84d232eee798a974ebaeb9c82d623f4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structrqs__time__type_1a9c5b9155e1665977103d8c32881d9f00>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structrqs__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structrqs__time__type_1a4c971b10c915041b89daa05a29125376>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structrqs__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_assemble<doxid-structrqs__time__type_1a4df2b92cea9269b8f8cad7024b83a10d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structrqs__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structrqs__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structrqs__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`;
	};
.. _details-structrqs__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structrqs__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total CPU time spent in the package

.. index:: pair: variable; assemble
.. _doxid-structrqs__time__type_1ae84d232eee798a974ebaeb9c82d623f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` assemble

CPU time spent building $H + \lambda M$.

.. index:: pair: variable; analyse
.. _doxid-structrqs__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

CPU time spent reordering $H + \lambda M$ prior to factorization.

.. index:: pair: variable; factorize
.. _doxid-structrqs__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

CPU time spent factorizing $H + \lambda M$.

.. index:: pair: variable; solve
.. _doxid-structrqs__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

CPU time spent solving linear systems inolving $H + \lambda M$.

.. index:: pair: variable; clock_total
.. _doxid-structrqs__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structrqs__time__type_1a4df2b92cea9269b8f8cad7024b83a10d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_assemble

clock time spent building $H + \lambda M$

.. index:: pair: variable; clock_analyse
.. _doxid-structrqs__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

clock time spent reordering $H + \lambda M$ prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structrqs__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

clock time spent factorizing $H + \lambda M$

.. index:: pair: variable; clock_solve
.. _doxid-structrqs__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

clock time spent solving linear systems inolving $H + \lambda M$

