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
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structbqp__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structbqp__time__type_1a0ca2b20748c7749a77d684124011c531>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structbqp__time__type_1ab7eecce4b013c87e490b8984c74c59c3>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structbqp__time__type_1a6356532c25755a6e5fedee1a7d703949>`;
	};
.. _details-structbqp__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbqp__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

total time

.. index:: pair: variable; analyse
.. _doxid-structbqp__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structbqp__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structbqp__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

time for the linear solution phase

