.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_scu_control_type.rst
	struct_scu_time_type.rst
	struct_scu_inform_type.rst

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
	struct_scu_control_type.rst
	struct_scu_inform_type.rst

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

	struct :ref:`scu_control_type<doxid-structscu__control__type>`;
	struct :ref:`scu_inform_type<doxid-structscu__inform__type>`;

	// global functions

	void :ref:`scu_information<doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d>`(void** data, struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform, int* status);

	void :ref:`scu_terminate<doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad>`(
		void** data,
		struct :ref:`scu_control_type<doxid-structscu__control__type>`* control,
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform
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

.. index:: pair: function; scu_information
.. _doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void scu_information(void** data, struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`scu_inform_type <doxid-structscu__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; scu_terminate
.. _doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void scu_terminate(
		void** data,
		struct :ref:`scu_control_type<doxid-structscu__control__type>`* control,
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`scu_control_type <doxid-structscu__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`scu_inform_type <doxid-structscu__inform__type>`)

