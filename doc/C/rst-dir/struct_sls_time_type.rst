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
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structsls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structsls__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structsls__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structsls__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`order_external<doxid-structsls__time__type_order_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse_external<doxid-structsls__time__type_analyse_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize_external<doxid-structsls__time__type_factorize_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve_external<doxid-structsls__time__type_solve_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structsls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structsls__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structsls__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structsls__time__type_clock_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_order_external<doxid-structsls__time__type_clock_order_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse_external<doxid-structsls__time__type_clock_analyse_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize_external<doxid-structsls__time__type_clock_factorize_external>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve_external<doxid-structsls__time__type_clock_solve_external>`;
	};
.. _details-structsls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structsls__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the total cpu time spent in the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structsls__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the total cpu time spent in the factorization phase

.. index:: pair: variable; solve
.. _doxid-structsls__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

the total cpu time spent in the solve phases

.. index:: pair: variable; order_external
.. _doxid-structsls__time__type_order_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` order_external

the total cpu time spent by the external solver in the ordering phase

.. index:: pair: variable; analyse_external
.. _doxid-structsls__time__type_analyse_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse_external

the total cpu time spent by the external solver in the analysis phase

.. index:: pair: variable; factorize_external
.. _doxid-structsls__time__type_factorize_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize_external

the total cpu time spent by the external solver in the factorization pha

.. index:: pair: variable; solve_external
.. _doxid-structsls__time__type_solve_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve_external

the total cpu time spent by the external solver in the solve phases

.. index:: pair: variable; clock_total
.. _doxid-structsls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structsls__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the total clock time spent in the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structsls__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the total clock time spent in the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structsls__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the total clock time spent in the solve phases

.. index:: pair: variable; clock_order_external
.. _doxid-structsls__time__type_clock_order_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_order_external

the total clock time spent by the external solver in the ordering phase

.. index:: pair: variable; clock_analyse_external
.. _doxid-structsls__time__type_clock_analyse_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse_external

the total clock time spent by the external solver in the analysis phase

.. index:: pair: variable; clock_factorize_external
.. _doxid-structsls__time__type_clock_factorize_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize_external

the total clock time spent by the external solver in the factorization p

.. index:: pair: variable; clock_solve_external
.. _doxid-structsls__time__type_clock_solve_external:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve_external

the total clock time spent by the external solver in the solve phases

