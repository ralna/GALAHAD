.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_llsr_control_type.rst
	struct_llsr_time_type.rst
	struct_llsr_history_type.rst
	struct_llsr_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`llsr_control_type<doxid-structllsr__control__type>`;
	struct :ref:`llsr_history_type<doxid-structllsr__history__type>`;
	struct :ref:`llsr_inform_type<doxid-structllsr__inform__type>`;
	struct :ref:`llsr_time_type<doxid-structllsr__time__type>`;

	// global functions

	void :ref:`llsr_initialize<doxid-galahad__llsr_8h_1a926f9824ab2e2bc71450a4d0b483879d>`(
		void** data,
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		int* status
	);

	void :ref:`llsr_read_specfile<doxid-galahad__llsr_8h_1a01bf988188420284ac4c125fef947efb>`(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		const char specfile[]
	);

	void :ref:`llsr_import<doxid-galahad__llsr_8h_1ac2dd0bee9270e6295c63a5365186070f>`(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char A_type[],
		int A_ne,
		const int A_row[],
		const int A_col[],
		const int A_ptr[]
	);

	void :ref:`llsr_import_scaling<doxid-galahad__llsr_8h_1a75f3108d65fc8100776af18f6adf4c2c>`(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status,
		int n,
		const char S_type[],
		int S_ne,
		const int S_row[],
		const int S_col[],
		const int S_ptr[]
	);

	void :ref:`llsr_reset_control<doxid-galahad__llsr_8h_1a9a9e3ae8ce66a5b7933b06061208c50c>`(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`llsr_solve_problem<doxid-galahad__llsr_8h_1aa702f5ffcde083da921921c4e1131d89>`(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		int A_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		int S_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` S_val[]
	);

	void :ref:`llsr_information<doxid-galahad__llsr_8h_1a1ba2eb99bc6364f476b85c7f507d43a2>`(void** data, struct :ref:`llsr_inform_type<doxid-structllsr__inform__type>`* inform, int* status);

	void :ref:`llsr_terminate<doxid-galahad__llsr_8h_1af05d27e76348279a8c9c16298a819609>`(
		void** data,
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		struct :ref:`llsr_inform_type<doxid-structllsr__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; real_sp_
.. _doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float real_sp_

``real_sp_`` is real single precision

.. index:: pair: typedef; real_wp_
.. _doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double real_wp_

``real_wp_`` is the real working precision used

function calls
--------------

.. index:: pair: function; llsr_initialize
.. _doxid-galahad__llsr_8h_1a926f9824ab2e2bc71450a4d0b483879d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_initialize(
		void** data,
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		int* status
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

		- is a struct containing control information (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; llsr_read_specfile
.. _doxid-galahad__llsr_8h_1a01bf988188420284ac4c125fef947efb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_read_specfile(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; llsr_import
.. _doxid-galahad__llsr_8h_1ac2dd0bee9270e6295c63a5365186070f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_import(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char A_type[],
		int A_ne,
		const int A_row[],
		const int A_col[],
		const int A_ptr[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was succesful, and the package is ready for the solve phase
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals, i.e., the number of rows of :math:`A`. m must be positive.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables, i.e., the number of columns of :math:`A`. n must be positive.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, :math:`A` if any. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in :math:`A`, if used, in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type int, that holds the row indices of :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of :math:`A` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of :math:`A`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; llsr_import_scaling
.. _doxid-galahad__llsr_8h_1a75f3108d65fc8100776af18f6adf4c2c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_import_scaling(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status,
		int n,
		const char S_type[],
		int S_ne,
		const int S_row[],
		const int S_col[],
		const int S_ptr[]
	)

Import the scaling matrix :math:`S` into internal storage prior to solution. Thus must have been preceeded by a call to llsr_import.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was succesful, and the package is ready for the solve phase
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows' or 'diagonal' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables, i.e., the number of rows and columns of :math:`S`. n must be positive.

	*
		- S_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid->` used for the matrix :math:`S`. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'diagonal'; lower or upper case variants are allowed.

	*
		- S_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of :math:`S` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- S_row

		- is a one-dimensional array of size S_ne and type int, that holds the row indices of the lower triangular part of :math:`S` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- S_col

		- is a one-dimensional array of size S_ne and type int, that holds the column indices of the lower triangular part of :math:`S` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- S_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of :math:`S`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; llsr_reset_control
.. _doxid-galahad__llsr_8h_1a9a9e3ae8ce66a5b7933b06061208c50c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_reset_control(
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; llsr_solve_problem
.. _doxid-galahad__llsr_8h_1aa702f5ffcde083da921921c4e1131d89:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_solve_problem(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		int A_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		int S_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` S_val[]
	)

Solve the regularization-region problem.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the entry and exit status from the package.
		  
		  Possible exit are:
		  
		  * 0. The run was succesful.
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m > 0 or requirement that A_type or A_type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows' or 'diagonal' has been violated.
		  
		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status
		  
		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.
		  
		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.
		  
		  * -15. The matrix :math:`S` does not appear to be strictly diagonally dominant.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.
		  
		  * -17. The step is too small to make further impact.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- power

		- is a scalar of type double, that holds the regularization power, :math:`p`, used. power must be greater than or equal to two.

	*
		- weight

		- is a scalar of type double, that holds the regularization power, :math:`\sigma`, used. weight must be strictly positive

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in the observation matrix :math:`A`.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type double, that holds the values of the entries of the observation matrix :math:`A` in any of the available storage schemes.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the values :math:`b` of observations. The i-th component of b, i = 0, ... , m-1, contains :math:`b_i`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- S_ne

		- is a scalar variable of type int, that holds the number of entries in the scaling matrix :math:`S` if it not the identity matrix.

	*
		- S_val

		- is a one-dimensional array of size S_ne and type double, that holds the values of the entries of the scaling matrix :math:`S` in any of the available storage schemes. If S_val is NULL, :math:`S` will be taken to be the identity matrix.

.. index:: pair: function; llsr_information
.. _doxid-galahad__llsr_8h_1a1ba2eb99bc6364f476b85c7f507d43a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_information(void** data, struct :ref:`llsr_inform_type<doxid-structllsr__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`llsr_inform_type <doxid-structllsr__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; llsr_terminate
.. _doxid-galahad__llsr_8h_1af05d27e76348279a8c9c16298a819609:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llsr_terminate(
		void** data,
		struct :ref:`llsr_control_type<doxid-structllsr__control__type>`* control,
		struct :ref:`llsr_inform_type<doxid-structllsr__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`llsr_control_type <doxid-structllsr__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`llsr_inform_type <doxid-structllsr__inform__type>`)

