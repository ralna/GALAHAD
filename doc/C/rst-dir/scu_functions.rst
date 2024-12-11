.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_scu_control_type.rst
	struct_scu_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`scu_control_type<doxid-structscu__control__type>`;
	struct :ref:`scu_inform_type<doxid-structscu__inform__type>`;

	// global functions

	void :ref:`scu_information<doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d>`(void **data, struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`scu_terminate<doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad>`(
		void **data,
		struct :ref:`scu_control_type<doxid-structscu__control__type>`* control,
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform
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

.. index:: pair: function; scu_information
.. _doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void scu_information(void **data, struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; scu_terminate
.. _doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void scu_terminate(
		void **data,
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

