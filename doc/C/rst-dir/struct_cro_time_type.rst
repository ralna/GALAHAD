.. index:: pair: struct; cro_time_type
.. _doxid-structcro__time__type:

cro_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_cro.h>
	
	struct cro_time_type {
		// components
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structcro__time__type_total>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structcro__time__type_analyse>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structcro__time__type_factorize>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structcro__time__type_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structcro__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structcro__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structcro__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structcro__time__type_clock_solve>`;
	};
.. _details-structcro__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structcro__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structcro__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

the CPU time spent reordering the matrix prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structcro__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; solve
.. _doxid-structcro__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

the CPU time spent computing corrections

.. index:: pair: variable; clock_total
.. _doxid-structcro__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structcro__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

the clock time spent analysing the required matrices prior to factorizat

.. index:: pair: variable; clock_factorize
.. _doxid-structcro__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

the clock time spent factorizing the required matrices

.. index:: pair: variable; clock_solve
.. _doxid-structcro__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

the clock time spent computing corrections

