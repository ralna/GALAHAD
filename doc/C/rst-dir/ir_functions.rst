.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_ir_control_type.rst
	struct_ir_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`ir_control_type<doxid-structir__control__type>`;
	struct :ref:`ir_inform_type<doxid-structir__inform__type>`;

	// global functions

	void :ref:`ir_initialize<doxid-galahad__ir_8h_1a1da2baeef0fe4c8e8937674a0c491c14>`(void** data, struct :ref:`ir_control_type<doxid-structir__control__type>`* control, int* status);
	void :ref:`ir_information<doxid-galahad__ir_8h_1a744223166561eeb7bf4f9e6d65d2e991>`(void** data, struct :ref:`ir_inform_type<doxid-structir__inform__type>`* inform, int* status);

	void :ref:`ir_terminate<doxid-galahad__ir_8h_1aa7a05607f24ba9e539d96c889aca1134>`(
		void** data,
		struct :ref:`ir_control_type<doxid-structir__control__type>`* control,
		struct :ref:`ir_inform_type<doxid-structir__inform__type>`* inform
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

.. index:: pair: function; ir_initialize
.. _doxid-galahad__ir_8h_1a1da2baeef0fe4c8e8937674a0c491c14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ir_initialize(void** data, struct :ref:`ir_control_type<doxid-structir__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`ir_control_type <doxid-structir__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; ir_information
.. _doxid-galahad__ir_8h_1a744223166561eeb7bf4f9e6d65d2e991:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ir_information(void** data, struct :ref:`ir_inform_type<doxid-structir__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`ir_inform_type <doxid-structir__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; ir_terminate
.. _doxid-galahad__ir_8h_1aa7a05607f24ba9e539d96c889aca1134:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ir_terminate(
		void** data,
		struct :ref:`ir_control_type<doxid-structir__control__type>`* control,
		struct :ref:`ir_inform_type<doxid-structir__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`ir_control_type <doxid-structir__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`ir_inform_type <doxid-structir__inform__type>`)

