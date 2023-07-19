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
	
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`total<doxid-structfdc__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`analyse<doxid-structfdc__time__type_1a9c5b9155e1665977103d8c32881d9f00>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`factorize<doxid-structfdc__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_total<doxid-structfdc__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_analyse<doxid-structfdc__time__type_1a3394e706afb175d930c81c4b86fe8f4b>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_factorize<doxid-structfdc__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
	};
.. _details-structfdc__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structfdc__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` total

the total CPU time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structfdc__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` analyse

the CPU time spent analysing the required matrices prior to factorization

.. index:: pair: variable; factorize
.. _doxid-structfdc__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` factorize

the CPU time spent factorizing the required matrices

.. index:: pair: variable; clock_total
.. _doxid-structfdc__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structfdc__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_analyse

the clock time spent analysing the required matrices prior to factorization

.. index:: pair: variable; clock_factorize
.. _doxid-structfdc__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_factorize

the clock time spent factorizing the required matrices

