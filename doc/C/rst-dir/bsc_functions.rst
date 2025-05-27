.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_bsc_control_type.rst
	struct_bsc_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>`;
	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`;

	// global functions

	void :ref:`bsc_initialize<doxid-galahad__bsc_initialize>`(void **data, struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`bsc_read_specfile<doxid-galahad__bsc_specfile>`(struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, const char specfile[]);
	void :ref:`bsc_import<doxid-galahad__bsc_import>`(
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne
	);

	void :ref:`bsc_reset_control<doxid-galahad__bsc_reset_control>`(
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`bsc_form_s<doxid-galahad__bsc_form_s>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_row[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ptr[],
		:ref:`ipc_<doxid-galahad__rpc_8h_>` S_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` D[]
	);
	void :ref:`bsc_information<doxid-galahad__bsc_information>`(void **data, struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`bsc_terminate<doxid-galahad__bsc_terminate>`(
		void **data,
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform
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

.. index:: pair: function; bsc_initialize
.. _doxid-galahad__bsc_initialize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_initialize(void **data, struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.


.. index:: pair: function; bsc_read_specfile
.. _doxid-galahad__bsc_specfile:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_read_specfile(struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/bsc/BSC.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/bsc.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; bsc_import
.. _doxid-galahad__bsc_import:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_import(
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne
	)

Import data into internal storage prior to solution and set up structure of $S$,

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful
		  
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
                    The restrictions n > 0 or m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate' or 'sparse_by_rows' has been violated.
		  
	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows of $A$.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of columns of $A$.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size m+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- S_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries required to hold $S$ in the sparse co-ordinate storage scheme.

.. index:: pair: function; bsc_reset_control
.. _doxid-galahad__bsc_reset_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_reset_control(
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful.

.. index:: pair: function; bsc_form_s
.. _doxid-galahad__bsc_form_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_form_s(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_row[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ptr[],
		:ref:`ipc_<doxid-galahad__rpc_8h_>` S_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` D[]
	)

Form the Schur complement matrix, $S$.


.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  Possible exit values are:
		  
		  * **0**
                    The run was successful.
		  
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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate' or 'sparse_by_rows' has been violated.
		  
	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows of $A$.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of columns of $A$.

	*
		- A_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $A$.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the matrix $A$ in any of the available storage schemes.

	*
		- S_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower traingle of $S$ in the sparse co-ordinate storage scheme.

	*
		- S_row

		- is a one-dimensional array of size S_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the row indices the lower traingle of $S$ in the sparse co-ordinate storage scheme.

	*
		- S_col

		- is a one-dimensional array of size S_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the column indices the lower traingle of $S$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. 

	*
		- S_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the starting position of each row the lower traingle of $S$, as well as the total number of entries, in the sparse row-wise storage scheme.

	*
		- S_val

		- is a one-dimensional array of size S_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that gives the values of the entries of the lower traingle of the matrix $S$.

	*
		- D

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that gives the values of the diagonal entries in $D$. If $D$ is the identity matrix, D can be NULL to save storage.


.. index:: pair: function; bsc_information
.. _doxid-galahad__bsc_information:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_information(void **data, struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; bsc_terminate
.. _doxid-galahad__bsc_terminate:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bsc_terminate(
		void **data,
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>`* control,
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

