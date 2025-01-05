.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_gls_control_type.rst
	struct_gls_ainfo_type.rst
	struct_gls_finfo_type.rst
	struct_gls_sinfo_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`gls_ainfo_type<doxid-structgls__ainfo>`;
	struct :ref:`gls_control_type<doxid-structgls__control__type>`;
	struct :ref:`gls_finfo_type<doxid-structgls__finfo>`;
	struct :ref:`gls_sinfo_type<doxid-structgls__sinfo>`;

	// global functions

	void :ref:`gls_initialize<doxid-galahad__gls_8h_1ab7827883517db347ee1229eda004ede5>`(void **data, struct :ref:`gls_control_type<doxid-structgls__control__type>`* control);
	void :ref:`gls_read_specfile<doxid-galahad__gls_8h_1a428c3dcc1d0de87f6108d396eec6e176>`(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, const char specfile[]);
	void :ref:`gls_import<doxid-galahad__gls_8h_1a1b34338e803f603af4161082a25f4e58>`(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`gls_reset_control<doxid-galahad__gls_8h_1a8b84f081ccc0b05b733adc2f0a829c07>`(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`gls_information<doxid-galahad__gls_8h_1a620dc0f7a9ef6049a7bafdc02913da47>`(
		void **data,
		struct :ref:`gls_ainfo_type<doxid-structgls__ainfo>`* ainfo,
		struct :ref:`gls_finfo_type<doxid-structgls__finfo>`* finfo,
		struct :ref:`gls_sinfo_type<doxid-structgls__sinfo>`* sinfo,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`gls_finalize<doxid-galahad__gls_8h_1a4758f1fc9cad110a33c1778254b51390>`(void **data, struct :ref:`gls_contro_typel<doxid-structgls__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

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

.. index:: pair: function; gls_initialize
.. _doxid-galahad__gls_8h_1ab7827883517db347ee1229eda004ede5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_initialize(void **data, struct :ref:`gls_control_type<doxid-structgls__control__type>`* control)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`gls_control_type <doxid-structgls__control__type>`)

.. index:: pair: function; gls_read_specfile
.. _doxid-galahad__gls_8h_1a428c3dcc1d0de87f6108d396eec6e176:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_read_specfile(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/gls/GLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/gls.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`gls_control_type <doxid-structgls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; gls_import
.. _doxid-galahad__gls_8h_1a1b34338e803f603af4161082a25f4e58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_import(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`gls_control_type <doxid-structgls__control__type>`)

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

.. index:: pair: function; gls_reset_control
.. _doxid-galahad__gls_8h_1a8b84f081ccc0b05b733adc2f0a829c07:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_reset_control(struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`gls_control_type <doxid-structgls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1** The import was successful, and the package is
                    ready for the solve phase

.. index:: pair: function; gls_information
.. _doxid-galahad__gls_8h_1a620dc0f7a9ef6049a7bafdc02913da47:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_information(
		void **data,
		struct :ref:`gls_ainfo_type<doxid-structgls__ainfo>`* ainfo,
		struct :ref:`gls_finfo_type<doxid-structgls__finfo>`* finfo,
		struct :ref:`gls_sinfo_type<doxid-structgls__sinfo>`* sinfo,
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

		- is a struct containing analysis output information (see :ref:`gls_ainfo_type <doxid-structgls__ainfo>`)

	*
		- finfo

		- is a struct containing factorization output information (see :ref:`gls_finfo_type <doxid-structgls__finfo>`)

	*
		- sinfo

		- is a struct containing solver output information (see :ref:`gls_sinfo_type <doxid-structgls__sinfo>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded successfully

.. index:: pair: function; gls_finalize
.. _doxid-galahad__gls_8h_1a4758f1fc9cad110a33c1778254b51390:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gls_finalize(void **data, struct :ref:`gls_control_type<doxid-structgls__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`gls_control_type <doxid-structgls__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

