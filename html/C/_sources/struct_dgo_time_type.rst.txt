.. index:: pair: struct; dgo_time_type
.. _doxid-structdgo__time__type:

dgo_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dgo.h>
	
	struct dgo_time_type {
		// components
	
		:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` :ref:`total<doxid-structdgo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`;
		:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` :ref:`univariate_global<doxid-structdgo__time__type_1ae803cab9cf49e3b9f259415e254f7a8e>`;
		:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` :ref:`multivariate_local<doxid-structdgo__time__type_1ae3473e3e6e1f5482c642784f7e5b85e7>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_total<doxid-structdgo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_univariate_global<doxid-structdgo__time__type_1a35fea348c7aed26574dec4efbd9a7107>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_multivariate_local<doxid-structdgo__time__type_1a7e6ac9410dc0d6af0a020612ad4fceb0>`;
	};
.. _details-structdgo__time__type:

detailed documentation
----------------------

time derived type as a C struct


components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structdgo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` total

the total CPU time spent in the package

.. index:: pair: variable; univariate_global
.. _doxid-structdgo__time__type_1ae803cab9cf49e3b9f259415e254f7a8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` univariate_global

the CPU time spent performing univariate global optimization

.. index:: pair: variable; multivariate_local
.. _doxid-structdgo__time__type_1ae3473e3e6e1f5482c642784f7e5b85e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` multivariate_local

the CPU time spent performing multivariate local optimization

.. index:: pair: variable; clock_total
.. _doxid-structdgo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_univariate_global
.. _doxid-structdgo__time__type_1a35fea348c7aed26574dec4efbd9a7107:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_univariate_global

the clock time spent performing univariate global optimization

.. index:: pair: variable; clock_multivariate_local
.. _doxid-structdgo__time__type_1a7e6ac9410dc0d6af0a020612ad4fceb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_multivariate_local

the clock time spent performing multivariate local optimization

