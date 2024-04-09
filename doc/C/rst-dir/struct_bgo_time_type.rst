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
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structbgo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`univariate_global<doxid-structbgo__time__type_1ae803cab9cf49e3b9f259415e254f7a8e>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`multivariate_local<doxid-structbgo__time__type_1ae3473e3e6e1f5482c642784f7e5b85e7>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structbgo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_univariate_global<doxid-structbgo__time__type_1a35fea348c7aed26574dec4efbd9a7107>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_multivariate_local<doxid-structbgo__time__type_1a7e6ac9410dc0d6af0a020612ad4fceb0>`;
	};
.. _details-structbgo__time__type:

detailed documentation
----------------------

time derived type as a C struct


components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbgo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; univariate_global
.. _doxid-structbgo__time__type_1ae803cab9cf49e3b9f259415e254f7a8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` univariate_global

the CPU time spent performing univariate global optimization

.. index:: pair: variable; multivariate_local
.. _doxid-structbgo__time__type_1ae3473e3e6e1f5482c642784f7e5b85e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` multivariate_local

the CPU time spent performing multivariate local optimization

.. index:: pair: variable; clock_total
.. _doxid-structbgo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_univariate_global
.. _doxid-structbgo__time__type_1a35fea348c7aed26574dec4efbd9a7107:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_univariate_global

the clock time spent performing univariate global optimization

.. index:: pair: variable; clock_multivariate_local
.. _doxid-structbgo__time__type_1a7e6ac9410dc0d6af0a020612ad4fceb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_multivariate_local

the clock time spent performing multivariate local optimization

