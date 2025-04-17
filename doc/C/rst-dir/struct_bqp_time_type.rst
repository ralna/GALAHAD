.. index:: pair: struct; bqp_time_type
.. _doxid-structbqp__time__type:

bqp_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bqp.h>
	
	struct bqp_time_type {
		// components
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structbqp__time__type_total>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structbqp__time__type_analyse>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structbqp__time__type_factorize>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structbqp__time__type_solve>`;
	};
.. _details-structbqp__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbqp__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

total time

.. index:: pair: variable; analyse
.. _doxid-structbqp__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structbqp__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structbqp__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

time for the linear solution phase

