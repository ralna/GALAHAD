.. index:: pair: struct; bgo_time_type
.. _doxid-structbgo__time__type:

bgo_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bgo.h>
	
	struct bgo_time_type {
		// components
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structbgo__time__type_total>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`univariate_global<doxid-structbgo__time__type_univariate_global>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`multivariate_local<doxid-structbgo__time__type_multivariate_local>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structbgo__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_univariate_global<doxid-structbgo__time__type_clock_univariate_global>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_multivariate_local<doxid-structbgo__time__type_clock_multivariate_local>`;
	};
.. _details-structbgo__time__type:

detailed documentation
----------------------

time derived type as a C struct


components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbgo__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; univariate_global
.. _doxid-structbgo__time__type_univariate_global:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` univariate_global

the CPU time spent performing univariate global optimization

.. index:: pair: variable; multivariate_local
.. _doxid-structbgo__time__type_multivariate_local:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` multivariate_local

the CPU time spent performing multivariate local optimization

.. index:: pair: variable; clock_total
.. _doxid-structbgo__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_univariate_global
.. _doxid-structbgo__time__type_clock_univariate_global:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_univariate_global

the clock time spent performing univariate global optimization

.. index:: pair: variable; clock_multivariate_local
.. _doxid-structbgo__time__type_clock_multivariate_local:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_multivariate_local

the clock time spent performing multivariate local optimization

