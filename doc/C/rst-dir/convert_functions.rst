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

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`convert_control_type<doxid-structconvert__control__type>`;
	struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`;
	struct :ref:`convert_time_type<doxid-structconvert__time__type>`;

	// global functions

	void :ref:`convert_initialize<doxid-galahad__convert_8h_1a6b9f4c72cb9c23cae9d4900816685ad6>`(
		void **data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`convert_information<doxid-galahad__convert_8h_1aba73d8119f28983efa000812bd970be6>`(
		void **data,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`convert_terminate<doxid-galahad__convert_8h_1a98d7b586061edf44052ae76b80cd2697>`(
		void **data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; spc_
.. _doxid-galahad__spc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float spc_

``spc_`` is real single precision

.. index:: pair: typedef; rpc_
.. _doxid-galahad__rpc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double rpc_

``rpc_`` is the real working precision used, but may be changed to ``float`` by
defining the  preprocessor variable ``SINGLE``.

.. index:: pair: typedef; ipc_
.. _doxid-galahad__ipc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int ipc_

``ipc_`` is the default integer word length used, but may be changed to 
``int64_t`` by defining the  preprocessor variable ``INTEGER_64``.

function calls
--------------

.. index:: pair: function; convert_initialize
.. _doxid-galahad__convert_8h_1a6b9f4c72cb9c23cae9d4900816685ad6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_initialize(
		void **data,
		struct :ref:`convert_control_type<doxid-structconvert__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; convert_information
.. _doxid-galahad__convert_8h_1aba73d8119f28983efa000812bd970be6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_information(
		void **data,
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>`* inform,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; convert_terminate
.. _doxid-galahad__convert_8h_1a98d7b586061edf44052ae76b80cd2697:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void convert_terminate(
		void **data,
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

