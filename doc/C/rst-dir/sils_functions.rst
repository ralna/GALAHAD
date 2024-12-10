.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sils_control_type.rst
	struct_sils_ainfo_type.rst
	struct_sils_finfo_type.rst
	struct_sils_sinfo_type.rst

.. _details-global:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>`;
	struct :ref:`sils_control_type<doxid-structsils__control__type>`;
	struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>`;
	struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>`;

	// global functions

	void :ref:`sils_initialize<doxid-galahad__sils_8h_1adfa46fc519194d9acfbeccac4c5a1af3>`(
		void **data,
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`sils_read_specfile<doxid-galahad__sils_8h_1a12447d25d91610c87b4c8ce7744aefd7>`(
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
		const char specfile[]
	);

	void :ref:`sils_import<doxid-galahad__sils_8h_1a78d5647031a8a4522541064853b021ba>`(struct :ref:`sils_control_type<doxid-structsils__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`sils_reset_control<doxid-galahad__sils_8h_1a34e5304b29c89525543cd512f426ac4f>`(
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`sils_information<doxid-galahad__sils_8h_1a27320b6d18c7508283cfb19dc8fecf37>`(
		void **data,
		struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>`* ainfo,
		struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>`* finfo,
		struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>`* sinfo,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`sils_finalize<doxid-galahad__sils_8h_1aa862612cd37fce35b1d35bd6ad295d82>`(void **data, struct :ref:`sils_control_type<doxid-structsils__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

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

.. index:: pair: function; sils_initialize
.. _doxid-galahad__sils_8h_1adfa46fc519194d9acfbeccac4c5a1af3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_initialize(
		void **data,
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
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

		- is a struct containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sils_read_specfile
.. _doxid-galahad__sils_8h_1a12447d25d91610c87b4c8ce7744aefd7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_read_specfile(
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list 
of keywords with associated default values is provided in 
\$GALAHAD/src/sils/SILS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/sils.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; sils_import
.. _doxid-galahad__sils_8h_1a78d5647031a8a4522541064853b021ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_import(struct :ref:`sils_control_type<doxid-structsils__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase
		  
		  * **-1**
                    An allocation error occurred. A message indicating
                    the offending array is written on unit
                    control.error, and the returned allocation status
                    and a string containing the name of the offending
                    array are held in inform.alloc_status and
                    inform.bad_alloc respectively.
		  
		  * **-2**
                    A deallocation error occurred. A message indicating
                    the offending array is written on unit control.error
                    and the returned allocation status and a string
                    containing the name of the offending array are held
                    in inform.alloc_status and inform.bad_alloc
                    respectively.
		  
		  * **-3**
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

.. index:: pair: function; sils_reset_control
.. _doxid-galahad__sils_8h_1a34e5304b29c89525543cd512f426ac4f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_reset_control(
		struct :ref:`sils_control_type<doxid-structsils__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; sils_information
.. _doxid-galahad__sils_8h_1a27320b6d18c7508283cfb19dc8fecf37:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_information(
		void **data,
		struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>`* ainfo,
		struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>`* finfo,
		struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>`* sinfo,
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
		- ainfo

		- is a struct containing output information (see :ref:`sils_ainfo_type <doxid-structsils__ainfo__type>`)

	*
		- finfo

		- is a struct containing output information (see :ref:`sils_finfo_type <doxid-structsils__finfo__type>`)

	*
		- sinfo

		- is a struct containing output information (see :ref:`sils_sinfo_type <doxid-structsils__sinfo__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sils_finalize
.. _doxid-galahad__sils_8h_1aa862612cd37fce35b1d35bd6ad295d82:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sils_finalize(void **data, struct :ref:`sils_control_type<doxid-structsils__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully
		  
		  * $\neq$ 0. The Fortran STAT value of an allocate or deallocate statement that has failed.

