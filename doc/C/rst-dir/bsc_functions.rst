.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_bsc_control_type.rst
	struct_bsc_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>`;
	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`;

	// global functions

	void :ref:`bsc_initialize<doxid-galahad__bsc_8h_1a32dd948f5ce268b0cdb340f435819c8e>`(void** data, struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, int* status);
	void :ref:`bsc_information<doxid-galahad__bsc_8h_1a4c0ae2150d39c240539e1d3be836b0af>`(void** data, struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform, int* status);

	void :ref:`bsc_terminate<doxid-galahad__bsc_8h_1a3a8a2f875e681225b4851d060e310271>`(
		void** data,
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; real_sp_
.. _doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float real_sp_

``real_sp_`` is real single precision

.. index:: pair: typedef; real_wp_
.. _doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double real_wp_

``real_wp_`` is the real working precision used

function calls
--------------

.. index:: pair: function; bsc_initialize
.. _doxid-galahad__bsc_8h_1a32dd948f5ce268b0cdb340f435819c8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_initialize(void** data, struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The initialization was successful.

.. index:: pair: function; bsc_information
.. _doxid-galahad__bsc_8h_1a4c0ae2150d39c240539e1d3be836b0af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_information(void** data, struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded successfully

.. index:: pair: function; bsc_terminate
.. _doxid-galahad__bsc_8h_1a3a8a2f875e681225b4851d060e310271:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_terminate(
		void** data,
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform
	)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

