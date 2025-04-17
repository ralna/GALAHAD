.. index:: pair: table; llst_time_type
.. _doxid-structllst__time__type:

llst_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_llst.h>
	
	struct llst_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structllst__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`assemble<doxid-structllst__time__type_assemble>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structllst__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structllst__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`solve<doxid-structllst__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structllst__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_assemble<doxid-structllst__time__type_clock_assemble>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structllst__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structllst__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structllst__time__type_clock_solve>`;
	};
.. _details-structllst__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structllst__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total CPU time spent in the package

.. index:: pair: variable; assemble
.. _doxid-structllst__time__type_assemble:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` assemble

CPU time assembling $K(\lambda)$ in (1)

.. index:: pair: variable; analyse
.. _doxid-structllst__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

CPU time spent analysing $K(\lambda)$.

.. index:: pair: variable; factorize
.. _doxid-structllst__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

CPU time spent factorizing $K(\lambda)$.

.. index:: pair: variable; solve
.. _doxid-structllst__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` solve

CPU time spent solving linear systems inolving $K(\lambda)$.

.. index:: pair: variable; clock_total
.. _doxid-structllst__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structllst__time__type_clock_assemble:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_assemble

clock time assembling $K(\lambda)$

.. index:: pair: variable; clock_analyse
.. _doxid-structllst__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

clock time spent analysing $K(\lambda)$

.. index:: pair: variable; clock_factorize
.. _doxid-structllst__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

clock time spent factorizing $K(\lambda)$

.. index:: pair: variable; clock_solve
.. _doxid-structllst__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

clock time spent solving linear systems inolving $K(\lambda)$

