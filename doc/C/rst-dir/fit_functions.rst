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

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`fit_control_type<doxid-structfit__control__type>`;
	struct :ref:`fit_inform_type<doxid-structfit__inform__type>`;

	// global functions

	void :ref:`fit_initialize<doxid-galahad__fit_8h_1a53019c0890b67dbc8c8efa541e652a73>`(void **data, struct :ref:`fit_control_type<doxid-structfit__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`fit_information<doxid-galahad__fit_8h_1addba9b8a0adf58039c6974a4d30af840>`(void **data, struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`fit_terminate<doxid-galahad__fit_8h_1a92659f3983205f2d9694e555b2439390>`(
		void **data,
		struct :ref:`fit_control_type<doxid-structfit__control__type>`* control,
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform
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

.. index:: pair: function; fit_initialize
.. _doxid-galahad__fit_8h_1a53019c0890b67dbc8c8efa541e652a73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_initialize(void **data, struct :ref:`fit_control_type<doxid-structfit__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; fit_information
.. _doxid-galahad__fit_8h_1addba9b8a0adf58039c6974a4d30af840:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_information(void **data, struct :ref:`fit_inform_type<doxid-structfit__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; fit_terminate
.. _doxid-galahad__fit_8h_1a92659f3983205f2d9694e555b2439390:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fit_terminate(
		void **data,
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

