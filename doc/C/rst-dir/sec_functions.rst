.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sec_control_type.rst
	struct_sec_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`sec_control_type<doxid-structsec__control__type>`;
	struct :ref:`sec_inform_type<doxid-structsec__inform__type>`;

	// global functions

	void :ref:`sec_initialize<doxid-galahad__sec_8h_1adf7e7f81c32214d1e79170023d5d47e5>`(struct :ref:`sec_control_type<doxid-structsec__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`sec_information<doxid-galahad__sec_8h_1a24da7562aed5f631b7c1e5442326f66e>`(void **data, struct :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`sec_terminate<doxid-galahad__sec_8h_1aff9c49fd2dea47f495445d0f357a8b19>`(
		void **data,
		struct :ref:`sec_control_type<doxid-structsec__control__type>`* control,
		struct :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform
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

.. index:: pair: function; sec_initialize
.. _doxid-galahad__sec_8h_1adf7e7f81c32214d1e79170023d5d47e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sec_initialize(struct :ref:`sec_control_type<doxid-structsec__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; sec_information
.. _doxid-galahad__sec_8h_1a24da7562aed5f631b7c1e5442326f66e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sec_information(void **data, struct :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`sec_inform_type <doxid-structsec__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sec_terminate
.. _doxid-galahad__sec_8h_1aff9c49fd2dea47f495445d0f357a8b19:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sec_terminate(
		void **data,
		struct :ref:`sec_control_type<doxid-structsec__control__type>`* control,
		struct :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`sec_inform_type <doxid-structsec__inform__type>`)

