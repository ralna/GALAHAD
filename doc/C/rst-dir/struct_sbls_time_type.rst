.. index:: pair: table; sbls_time_type
.. _doxid-structsbls__time__type:

sbls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sbls.h>
	
	struct sbls_time_type {
		// fields
	
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`total<doxid-structsbls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`form<doxid-structsbls__time__type_1a8ac63de5e103d8e01b0e0f88bb7d230d>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`factorize<doxid-structsbls__time__type_1a79e62dbb4cbb6e99d82167e60c703015>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`apply<doxid-structsbls__time__type_1a9d8129bf5b1a9f21dfcc24dc5c706274>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_total<doxid-structsbls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_form<doxid-structsbls__time__type_1ab275f3b71b8e019aa35acf43c3fd7473>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_factorize<doxid-structsbls__time__type_1ad3f0f50628260b90d6cf974e02f86192>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_apply<doxid-structsbls__time__type_1afbbb1dd5fc63c640620fbd32a0481493>`;
	};
.. _details-structsbls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsbls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` total

total cpu time spent in the package

.. index:: pair: variable; form
.. _doxid-structsbls__time__type_1a8ac63de5e103d8e01b0e0f88bb7d230d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` form

cpu time spent forming the preconditioner $K_G$

.. index:: pair: variable; factorize
.. _doxid-structsbls__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` factorize

cpu time spent factorizing $K_G$

.. index:: pair: variable; apply
.. _doxid-structsbls__time__type_1a9d8129bf5b1a9f21dfcc24dc5c706274:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` apply

cpu time spent solving linear systems inolving $K_G$

.. index:: pair: variable; clock_total
.. _doxid-structsbls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_form
.. _doxid-structsbls__time__type_1ab275f3b71b8e019aa35acf43c3fd7473:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_form

clock time spent forming the preconditioner $K_G$

.. index:: pair: variable; clock_factorize
.. _doxid-structsbls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_factorize

clock time spent factorizing $K_G$

.. index:: pair: variable; clock_apply
.. _doxid-structsbls__time__type_1afbbb1dd5fc63c640620fbd32a0481493:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_apply

clock time spent solving linear systems inolving $K_G$

