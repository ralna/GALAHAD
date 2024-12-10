.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_lms_control_type.rst
	struct_lms_time_type.rst
	struct_lms_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`lms_control_type<doxid-structlms__control__type>`;
	struct :ref:`lms_inform_type<doxid-structlms__inform__type>`;
	struct :ref:`lms_time_type<doxid-structlms__time__type>`;

	// global functions

	void :ref:`lms_initialize<doxid-galahad__lms_8h_1a9abec0f0f82474e01c99ce43ab9252f5>`(void **data, struct :ref:`lms_control_type<doxid-structlms__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`lms_information<doxid-galahad__lms_8h_1a0c692aa607e53b87fd8a1a8de116f5aa>`(void **data, struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`lms_terminate<doxid-galahad__lms_8h_1a6c036818c80d8e54dcf4d0e7bb341e33>`(
		void **data,
		struct :ref:`lms_control_type<doxid-structlms__control__type>`* control,
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform
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
defining the  preprocessor variable ``REAL_32`` or (if supported) to
``__real128`` using the variable ``REAL_128``.

.. index:: pair: typedef; ipc_
.. _doxid-galahad__ipc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int ipc_

``ipc_`` is the default integer word length used, but may be changed to 
``int64_t`` by defining the  preprocessor variable ``INTEGER_64``.

function calls
--------------

.. index:: pair: function; lms_initialize
.. _doxid-galahad__lms_8h_1a9abec0f0f82474e01c99ce43ab9252f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_initialize(void **data, struct :ref:`lms_control_type<doxid-structlms__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; lms_information
.. _doxid-galahad__lms_8h_1a0c692aa607e53b87fd8a1a8de116f5aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_information(void **data, struct :ref:`lms_inform_type<doxid-structlms__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lms_terminate
.. _doxid-galahad__lms_8h_1a6c036818c80d8e54dcf4d0e7bb341e33:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lms_terminate(
		void **data,
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

