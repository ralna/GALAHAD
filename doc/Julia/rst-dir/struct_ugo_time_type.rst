.. index:: pair: struct; ugo_time_type
.. _doxid-structugo__time__type:

ugo_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ugo.h>
	
	struct ugo_time_type {
		// components
	
		:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` :ref:`total<doxid-structugo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_total<doxid-structugo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
	};
.. _details-structugo__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structugo__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>` total

the total CPU time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structugo__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_total

the total clock time spent in the package

