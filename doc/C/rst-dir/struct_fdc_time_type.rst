.. index:: pair: table; fdc_time_type
.. _doxid-structfdc__time__type:

fdc_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_fdc.h>
	
	struct fdc_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structfdc__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`analyse<doxid-structfdc__time__type_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structfdc__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structfdc__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structfdc__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structfdc__time__type_clock_factorize>`;
	};
.. _details-structfdc__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structfdc__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structfdc__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structfdc__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; clock_total
.. _doxid-structfdc__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structfdc__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structfdc__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

