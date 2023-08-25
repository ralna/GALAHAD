.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_fit_control_type.rst
	struct_fit_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`fit_control_type<doxid-structfit__control__type>`;
	struct :ref:`fit_inform_type<doxid-structfit__inform__type>`;

	// global functions

	void :ref:`fit_initialize<doxid-galahad__fit_8h_1a53019c0890b67dbc8c8efa541e652a73>`(void** data, struct :ref:`fit_control_type<doxid-structfit__control__type>`* control, int* status);
	void :ref:`fit_information<doxid-galahad__fit_8h_1addba9b8a0adf58039c6974a4d30af840>`(void** data, struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform, int* status);

	void :ref:`fit_terminate<doxid-galahad__fit_8h_1a92659f3983205f2d9694e555b2439390>`(
		void** data,
		struct :ref:`fit_control_type<doxid-structfit__control__type>`* control,
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform
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

.. index:: pair: function; fit_initialize
.. _doxid-galahad__fit_8h_1a53019c0890b67dbc8c8efa541e652a73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_initialize(void** data, struct :ref:`fit_control_type<doxid-structfit__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`fit_control_type <doxid-structfit__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; fit_information
.. _doxid-galahad__fit_8h_1addba9b8a0adf58039c6974a4d30af840:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_information(void** data, struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`fit_inform_type <doxid-structfit__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; fit_terminate
.. _doxid-galahad__fit_8h_1a92659f3983205f2d9694e555b2439390:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_terminate(
		void** data,
		struct :ref:`fit_control_type<doxid-structfit__control__type>`* control,
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`fit_control_type <doxid-structfit__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`fit_inform_type <doxid-structfit__inform__type>`)

