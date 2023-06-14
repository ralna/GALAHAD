.. index:: pair: table; bsc_inform_type
.. _doxid-structbsc__inform__type:

bsc_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. _details-structbsc__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: struct; bsc_inform_type
.. _doxid-structbsc__inform__type:

struct bsc_inform_type
======================

.. toctree::
	:hidden:

Overview
~~~~~~~~

inform derived type as a C struct :ref:`More...<details-structbsc__inform__type>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bsc.h>
	
	struct bsc_inform_type {
		// fields
	
		int :ref:`status<doxid-structbsc__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		int :ref:`alloc_status<doxid-structbsc__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structbsc__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		int :ref:`max_col_a<doxid-structbsc__inform__type_1afa7a035a987c746be5967674390199c2>`;
		int :ref:`exceeds_max_col<doxid-structbsc__inform__type_1a182eb2982f975bbbdce953a362bf0c2d>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`time<doxid-structbsc__inform__type_1afb418c91547eccd73ea85693be0e1cea>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`clock_time<doxid-structbsc__inform__type_1a70a6a867f00d49b434e314b7177c6f23>`;
	};
.. _details-structbsc__inform__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

inform derived type as a C struct

Fields
------

.. index:: pair: variable; status
.. _doxid-structbsc__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int status

the return status from the package. Possible values are:

* 0. The call was succesful

* -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

* -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

* -3. The restrictions n > 0 or m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.

.. index:: pair: variable; alloc_status
.. _doxid-structbsc__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbsc__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; max_col_a
.. _doxid-structbsc__inform__type_1afa7a035a987c746be5967674390199c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int max_col_a

the maximum number of entries in a column of :math:`A`

.. index:: pair: variable; exceeds_max_col
.. _doxid-structbsc__inform__type_1a182eb2982f975bbbdce953a362bf0c2d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int exceeds_max_col

the number of columns of :math:`A` that have more than control.max_col entries

.. index:: pair: variable; time
.. _doxid-structbsc__inform__type_1afb418c91547eccd73ea85693be0e1cea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` time

the total CPU time spent in the package

.. index:: pair: variable; clock_time
.. _doxid-structbsc__inform__type_1a70a6a867f00d49b434e314b7177c6f23:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` clock_time

the total clock time spent in the package

