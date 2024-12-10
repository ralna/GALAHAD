.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_hash_control_type.rst
	struct_hash_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`hash_control_type<doxid-structhash__control__type>`;
	struct :ref:`hash_inform_type<doxid-structhash__inform__type>`;

	// global functions

	void :ref:`hash_initialize<doxid-galahad__hash_8h_1ac983b0236ce2f2ae9ed016846c5ad2a3>`(
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nchar,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` length,
		void **data,
		struct :ref:`hash_control_type<doxid-structhash__control__type>`* control,
		struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform
	);

	void :ref:`hash_information<doxid-galahad__hash_8h_1a7f73a5ca2bbdc3af1b7793f7b14ed13f>`(void **data, struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`hash_terminate<doxid-galahad__hash_8h_1a0aece137337307f3c98e9b201205170d>`(
		void **data,
		struct :ref:`hash_control_type<doxid-structhash__control__type>`* control,
		struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform
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

.. index:: pair: function; hash_initialize
.. _doxid-galahad__hash_8h_1ac983b0236ce2f2ae9ed016846c5ad2a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void hash_initialize(
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nchar,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` length,
		void **data,
		struct :ref:`hash_control_type<doxid-structhash__control__type>`* control,
		struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform
	)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- nchar

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of characters permitted in each word in the hash table

	*
		- length

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the maximum number of words that can be held in the dictionary

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`hash_control_type <doxid-structhash__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)

.. index:: pair: function; hash_information
.. _doxid-galahad__hash_8h_1a7f73a5ca2bbdc3af1b7793f7b14ed13f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void hash_information(void **data, struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; hash_terminate
.. _doxid-galahad__hash_8h_1a0aece137337307f3c98e9b201205170d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void hash_terminate(
		void **data,
		struct :ref:`hash_control_type<doxid-structhash__control__type>`* control,
		struct :ref:`hash_inform_type<doxid-structhash__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`hash_control_type <doxid-structhash__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)

