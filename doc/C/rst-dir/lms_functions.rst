.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_lms_control_type.rst
	struct_lms_time_type.rst
	struct_lms_inform_type.rst

.. _details-global:

function calls
--------------

.. ---------------------------------------------------------------------------
.. _global:
.. index:: pair: namespace; global

Global Namespace
================

.. toctree::
	:hidden:

	namespace_conf.rst
	struct_lms_control_type.rst
	struct_lms_inform_type.rst
	struct_lms_time_type.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`lms_control_type<doxid-structlms__control__type>`;
	struct :ref:`lms_inform_type<doxid-structlms__inform__type>`;
	struct :ref:`lms_time_type<doxid-structlms__time__type>`;

	// global functions

	void :ref:`lms_initialize<doxid-galahad__lms_8h_1a9abec0f0f82474e01c99ce43ab9252f5>`(void** data, struct :ref:`lms_control_type<doxid-structlms__control__type>`* control, int* status);
	void :ref:`lms_information<doxid-galahad__lms_8h_1a0c692aa607e53b87fd8a1a8de116f5aa>`(void** data, struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform, int* status);

	void :ref:`lms_terminate<doxid-galahad__lms_8h_1a6c036818c80d8e54dcf4d0e7bb341e33>`(
		void** data,
		struct :ref:`lms_control_type<doxid-structlms__control__type>`* control,
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform
	);

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Typedefs
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

Global Functions
----------------

.. index:: pair: function; lms_initialize
.. _doxid-galahad__lms_8h_1a9abec0f0f82474e01c99ce43ab9252f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_initialize(void** data, struct :ref:`lms_control_type<doxid-structlms__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`lms_control_type <doxid-structlms__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The initialization was succesful.

.. index:: pair: function; lms_information
.. _doxid-galahad__lms_8h_1a0c692aa607e53b87fd8a1a8de116f5aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_information(void** data, struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`lms_inform_type <doxid-structlms__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; lms_terminate
.. _doxid-galahad__lms_8h_1a6c036818c80d8e54dcf4d0e7bb341e33:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_terminate(
		void** data,
		struct :ref:`lms_control_type<doxid-structlms__control__type>`* control,
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`lms_control_type <doxid-structlms__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`lms_inform_type <doxid-structlms__inform__type>`)

