.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_roots_control_type.rst
	struct_roots_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`roots_control_type<doxid-structroots__control__type>`;
	struct :ref:`roots_inform_type<doxid-structroots__inform__type>`;

	// global functions

	void :ref:`roots_initialize<doxid-galahad__roots_8h_1ac504c30e3b55ef425516ad7cf1638a76>`(
		void **data,
		struct :ref:`roots_control_type<doxid-structroots__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`roots_information<doxid-galahad__roots_8h_1ac63ef195952ae821d6966a8f25ac2513>`(
		void **data,
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>`* inform,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`roots_terminate<doxid-galahad__roots_8h_1a4e81f4ac6c1119dfeb3a81729c3ec997>`(
		void **data,
		struct :ref:`roots_control_type<doxid-structroots__control__type>`* control,
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>`* inform
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

.. index:: pair: function; roots_initialize
.. _doxid-galahad__roots_8h_1ac504c30e3b55ef425516ad7cf1638a76:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void roots_initialize(
		void **data,
		struct :ref:`roots_control_type<doxid-structroots__control__type>`* control,
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

		- is a struct containing control information (see :ref:`roots_control_type <doxid-structroots__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; roots_information
.. _doxid-galahad__roots_8h_1ac63ef195952ae821d6966a8f25ac2513:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void roots_information(
		void **data,
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>`* inform,
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

		- is a struct containing output information (see :ref:`roots_inform_type <doxid-structroots__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; roots_terminate
.. _doxid-galahad__roots_8h_1a4e81f4ac6c1119dfeb3a81729c3ec997:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void roots_terminate(
		void **data,
		struct :ref:`roots_control_type<doxid-structroots__control__type>`* control,
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`roots_control_type <doxid-structroots__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`roots_inform_type <doxid-structroots__inform__type>`)

