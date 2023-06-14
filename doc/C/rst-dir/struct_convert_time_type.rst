.. index:: pair: table; convert_time_type
.. _doxid-structconvert__time__type:

convert_time_type structure
-----------------------

.. toctree::
	:hidden:

.. _details-structconvert__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: struct; convert_time_type
.. _doxid-structconvert__time__type:

struct convert_time_type
========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

time derived type as a C struct :ref:`More...<details-structconvert__time__type>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_convert.h>
	
	struct convert_time_type {
		// fields
	
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`total<doxid-structconvert__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_total<doxid-structconvert__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
	};
.. _details-structconvert__time__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

time derived type as a C struct

Fields
------

.. index:: pair: variable; total
.. _doxid-structconvert__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` total

total cpu time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structconvert__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_total

total clock time spent in the package

