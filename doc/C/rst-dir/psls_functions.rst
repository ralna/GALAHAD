.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_psls_control_type.rst
	struct_psls_time_type.rst
	struct_psls_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`psls_control_type<doxid-structpsls__control__type>`;
	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>`;
	struct :ref:`psls_time_type<doxid-structpsls__time__type>`;

	// global functions

	void :ref:`psls_initialize<doxid-galahad__psls_8h_1af5cb66dbf5b9e4f094e2e0a29631fd1b>`(
		void **data,
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`psls_read_specfile<doxid-galahad__psls_8h_1a34b978446b6aa5636f9e6efc18860366>`(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		const char specfile[]
	);

	void :ref:`psls_import<doxid-galahad__psls_8h_1a3ff902c85fb82f1929a93514bb63c5d6>`(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` ptr[]
	);

	void :ref:`psls_reset_control<doxid-galahad__psls_8h_1a90493b62c689237c97fe4aea665cd0ab>`(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`psls_form_preconditioner<doxid-galahad__psls_8h_1a9cd4c449dcc5133932972866fd58cfc1>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[]
	);

	void :ref:`psls_form_subset_preconditioner<doxid-galahad__psls_8h_1a75fa79fcbe08ab367b9fa0b7f39adf65>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_sub,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` sub[]
	);

	void :ref:`psls_update_preconditioner<doxid-galahad__psls_8h_1a42a8097e64b527cff18ab66c07a32d1d>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_del,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` del[]
	);

	void :ref:`psls_apply_preconditioner<doxid-galahad__psls_8h_1a1bae97d4a0e63bce7380422ed83306e8>`(void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status, :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`rpc_<doxid-galahad__rpc_8h_>` sol[]);
	void :ref:`psls_information<doxid-galahad__psls_8h_1ace5f302a9ccb0c3f8c29b28b42da7793>`(void **data, struct :ref:`psls_inform_type<doxid-structpsls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`psls_terminate<doxid-galahad__psls_8h_1ab62a2e262e7466fac3a2dc8cd300720d>`(
		void **data,
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>`* inform
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

.. index:: pair: function; psls_initialize
.. _doxid-galahad__psls_8h_1af5cb66dbf5b9e4f094e2e0a29631fd1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_initialize(
		void **data,
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
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

		- is a struct containing control information (see :ref:`psls_control_type <doxid-structpsls__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; psls_read_specfile
.. _doxid-galahad__psls_8h_1a34b978446b6aa5636f9e6efc18860366:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_read_specfile(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list 
of keywords with associated default values is provided in 
\$GALAHAD/src/psls/PSLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/psls.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`psls_control_type <doxid-structpsls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; psls_import
.. _doxid-galahad__psls_8h_1a3ff902c85fb82f1929a93514bb63c5d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_import(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` ptr[]
	)

Import structural matrix data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`psls_control_type <doxid-structpsls__control__type>`)

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
                    'sparse_by_rows' or 'diagonal' has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows in the symmetric matrix $A$.

	*
		- type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- row

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- col

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense storage scheme is used, and in this case can be NULL.

	*
		- ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; psls_reset_control
.. _doxid-galahad__psls_8h_1a90493b62c689237c97fe4aea665cd0ab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_reset_control(
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`psls_control_type <doxid-structpsls__control__type>`)

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

.. index:: pair: function; psls_form_preconditioner
.. _doxid-galahad__psls_8h_1a9cd4c449dcc5133932972866fd58cfc1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_form_preconditioner(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[]
	)

Form and factorize a preconditioner $P$ of the matrix $A$.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * **0**
                    The factors were generated successfully.
		  
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
		  
		  * **-26**
                    The requested solver is not available.
		  
		  * **-29**
                    This option is not available with this solver.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the symmetric matrix $A$.

	*
		- val

		- is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the symmetric matrix $A$ in any of the supported storage schemes.

.. index:: pair: function; psls_form_subset_preconditioner
.. _doxid-galahad__psls_8h_1a75fa79fcbe08ab367b9fa0b7f39adf65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_form_subset_preconditioner(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_sub,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` sub[]
	)

Form and factorize a $P$ preconditioner of a symmetric submatrix of the matrix $A$.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * **0**
                    The factors were generated successfully.
		  
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
		  
		  * **-26**
                    The requested solver is not available.
		  
		  * **-29**
                    This option is not available with this solver.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the symmetric matrix $A$.

	*
		- val

		- is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the symmetric matrix $A$ in any of the supported storage schemes.

	*
		- n_sub

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows (and columns) of the required submatrix of $A$.

	*
		- sub

		- is a one-dimensional array of size n_sub and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the indices of the rows of required submatrix.

.. index:: pair: function; psls_update_preconditioner
.. _doxid-galahad__psls_8h_1a42a8097e64b527cff18ab66c07a32d1d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_update_preconditioner(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_del,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` del[]
	)

Update the preconditioner $P$ when rows (amd columns) are removed.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * **0**
                    The factors were generated successfully.
		  
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
		  
		  * **-26**
                    The requested solver is not available.
		  
		  * **-29**
                    This option is not available with this solver.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the symmetric matrix $A$.

	*
		- val

		- is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the symmetric matrix $A$ in any of the supported storage schemes.

	*
		- n_del

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows (and columns) of (sub) matrix that are to be deleted.

	*
		- del

		- is a one-dimensional array of size n_fix and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the indices of the rows that are to be deleted.

.. index:: pair: function; psls_apply_preconditioner
.. _doxid-galahad__psls_8h_1a1bae97d4a0e63bce7380422ed83306e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_apply_preconditioner(void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status, :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`rpc_<doxid-galahad__rpc_8h_>` sol[])

Solve the linear system $Px=b$.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * **0**
                    The required solution was obtained.
		  
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

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the vectors $b$ and $x$.

	*
		- sol

		- is a one-dimensional array of size n and type double. On entry, it must hold the vector $b$. On a successful exit, its contains the solution $x$. Any component corresponding to rows/columns not in the initial subset recorded by psls_form_subset_preconditioner, or in those subsequently deleted by psls_update_preconditioner, will not be altered.

.. index:: pair: function; psls_information
.. _doxid-galahad__psls_8h_1ace5f302a9ccb0c3f8c29b28b42da7793:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_information(void **data, struct :ref:`psls_inform_type<doxid-structpsls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provide output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`psls_inform_type <doxid-structpsls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; psls_terminate
.. _doxid-galahad__psls_8h_1ab62a2e262e7466fac3a2dc8cd300720d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void psls_terminate(
		void **data,
		struct :ref:`psls_control_type<doxid-structpsls__control__type>`* control,
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`psls_control_type <doxid-structpsls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`psls_inform_type <doxid-structpsls__inform__type>`)

