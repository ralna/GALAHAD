.. index:: pair: table; ssls_time_type
.. _doxid-structssls__time__type:

ssls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ssls.h>
	
	struct ssls_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structssls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structssls__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structssls__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structssls__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structssls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structssls__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structssls__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structssls__time__type_clock_solve>`;
	};
.. _details-structssls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structssls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structssls__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

cpu time spent forming, and analysing the structure of, $K$

.. index:: pair: variable; factorize
.. _doxid-structssls__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

cpu time spent factorizing $K$

.. index:: pair: variable; solve
.. _doxid-structssls__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

cpu time spent solving linear systems inolving $K$

.. index:: pair: variable; clock_total
.. _doxid-structssls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structssls__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

clock time spent forming, and analysing the structure of, $K$

.. index:: pair: variable; clock_factorize
.. _doxid-structssls__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

clock time spent factorizing $K$

.. index:: pair: variable; clock_solve
.. _doxid-structssls__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

clock time spent solving linear systems inolving $K$

