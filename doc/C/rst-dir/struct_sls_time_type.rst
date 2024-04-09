.. index:: pair: table; sls_time_type
.. _doxid-structsls__time__type:

sls_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sls.h>
	
	struct sls_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structsls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structsls__time__type_1a9c5b9155e1665977103d8c32881d9f00>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structsls__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structsls__time__type_1a4c971b10c915041b89daa05a29125376>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`order_external<doxid-structsls__time__type_1ac9e0db16df3b373e6192360c84f71aab>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse_external<doxid-structsls__time__type_1a9b4f1f33374092b099ab7baad8d1d6ac>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize_external<doxid-structsls__time__type_1a87f0b3565b139ff7baa76ce830a92964>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve_external<doxid-structsls__time__type_1a8cbd0409a559f5e24c15591cf8d53eeb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structsls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structsls__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structsls__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structsls__time__type_1af569df4b8828eb7ac8a05ef1030d1358>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_order_external<doxid-structsls__time__type_1a7babcea658f1454261df6b8acc24be9b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse_external<doxid-structsls__time__type_1ac479ec45cbef59b5aff36cc55861dc63>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize_external<doxid-structsls__time__type_1aab3765fc7d7de7a6148eb861ebd8fa31>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve_external<doxid-structsls__time__type_1abcd8605d13ed5379a535b305c69ee1cc>`;
	};
.. _details-structsls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structsls__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the total cpu time spent in the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structsls__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the total cpu time spent in the factorization phase

.. index:: pair: variable; solve
.. _doxid-structsls__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

the total cpu time spent in the solve phases

.. index:: pair: variable; order_external
.. _doxid-structsls__time__type_1ac9e0db16df3b373e6192360c84f71aab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` order_external

the total cpu time spent by the external solver in the ordering phase

.. index:: pair: variable; analyse_external
.. _doxid-structsls__time__type_1a9b4f1f33374092b099ab7baad8d1d6ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse_external

the total cpu time spent by the external solver in the analysis phase

.. index:: pair: variable; factorize_external
.. _doxid-structsls__time__type_1a87f0b3565b139ff7baa76ce830a92964:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize_external

the total cpu time spent by the external solver in the factorization pha

.. index:: pair: variable; solve_external
.. _doxid-structsls__time__type_1a8cbd0409a559f5e24c15591cf8d53eeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve_external

the total cpu time spent by the external solver in the solve phases

.. index:: pair: variable; clock_total
.. _doxid-structsls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structsls__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the total clock time spent in the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structsls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the total clock time spent in the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structsls__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the total clock time spent in the solve phases

.. index:: pair: variable; clock_order_external
.. _doxid-structsls__time__type_1a7babcea658f1454261df6b8acc24be9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_order_external

the total clock time spent by the external solver in the ordering phase

.. index:: pair: variable; clock_analyse_external
.. _doxid-structsls__time__type_1ac479ec45cbef59b5aff36cc55861dc63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse_external

the total clock time spent by the external solver in the analysis phase

.. index:: pair: variable; clock_factorize_external
.. _doxid-structsls__time__type_1aab3765fc7d7de7a6148eb861ebd8fa31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize_external

the total clock time spent by the external solver in the factorization p

.. index:: pair: variable; clock_solve_external
.. _doxid-structsls__time__type_1abcd8605d13ed5379a535b305c69ee1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve_external

the total clock time spent by the external solver in the solve phases

