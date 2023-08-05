.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sha_control_type.rst
	struct_sha_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`sha_control_type<doxid-structsha__control__type>`;
	struct :ref:`sha_inform_type<doxid-structsha__inform__type>`;

	// global functions

	void :ref:`sha_initialize<doxid-galahad__sha_8h_1aa4f01a598c5cef45420937d4951519a9>`(void** data, struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, int* status);
	void :ref:`sha_information<doxid-galahad__sha_8h_1ace357a88ad000654a0ec0817d6d28ece>`(void** data, struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform, int* status);

	void :ref:`sha_terminate<doxid-galahad__sha_8h_1a9ea67bcd115e6479e7d93faf3445405a>`(
		void** data,
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform
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

.. index:: pair: function; sha_initialize
.. _doxid-galahad__sha_8h_1aa4f01a598c5cef45420937d4951519a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_initialize(void** data, struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The initialization was successful.

.. index:: pair: function; sha_information
.. _doxid-galahad__sha_8h_1ace357a88ad000654a0ec0817d6d28ece:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_information(void** data, struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`sha_inform_type <doxid-structsha__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded successfully

.. index:: pair: function; sha_terminate
.. _doxid-galahad__sha_8h_1a9ea67bcd115e6479e7d93faf3445405a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_terminate(
		void** data,
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`sha_inform_type <doxid-structsha__inform__type>`)

