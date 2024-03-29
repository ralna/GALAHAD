.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_convert_control_type.rst
	struct_convert_time_type.rst
	struct_convert_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`convert_control_type<doxid-structconvert__control__type>`;
	struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`;
	struct :ref:`convert_time_type<doxid-structconvert__time__type>`;

	// global functions

	void :ref:`convert_initialize<doxid-galahad__convert_8h_1a6b9f4c72cb9c23cae9d4900816685ad6>`(
		void** data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		int* status
	);

	void :ref:`convert_information<doxid-galahad__convert_8h_1aba73d8119f28983efa000812bd970be6>`(
		void** data,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform,
		int* status
	);

	void :ref:`convert_terminate<doxid-galahad__convert_8h_1a98d7b586061edf44052ae76b80cd2697>`(
		void** data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform
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

.. index:: pair: function; convert_initialize
.. _doxid-galahad__convert_8h_1a6b9f4c72cb9c23cae9d4900816685ad6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_initialize(
		void** data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		int* status
	)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`convert_control_type <doxid-structconvert__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; convert_information
.. _doxid-galahad__convert_8h_1aba73d8119f28983efa000812bd970be6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_information(
		void** data,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform,
		int* status
	)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`convert_inform_type <doxid-structconvert__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; convert_terminate
.. _doxid-galahad__convert_8h_1a98d7b586061edf44052ae76b80cd2697:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_terminate(
		void** data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`convert_control_type <doxid-structconvert__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`convert_inform_type <doxid-structconvert__inform__type>`)

