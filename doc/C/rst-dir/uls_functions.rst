.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_uls_control_type.rst
	struct_uls_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`uls_control_type<doxid-structuls__control__type>`;
	struct :ref:`uls_inform_type<doxid-structuls__inform__type>`;

	// global functions

	void :ref:`uls_initialize<doxid-galahad__uls_8h_1a7afb5f2dde112e60686a5527a8f37ca4>`(
		const char solver[],
		void** data,
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		int* status
	);

	void :ref:`uls_read_specfile<doxid-galahad__uls_8h_1a5e2c9573bc8661114e9f073782b460ef>`(struct :ref:`uls_control_type<doxid-structuls__control__type>`* control, const char specfile[]);

	void :ref:`uls_factorize_matrix<doxid-galahad__uls_8h_1a6c0599479b84ee7d7c4ee7c473b76a83>`(
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char type[],
		int ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` val[],
		const int row[],
		const int col[],
		const int ptr[]
	);

	void :ref:`uls_reset_control<doxid-galahad__uls_8h_1ad2ad6daa4d54d75e40fbe253f2bc5881>`(
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`uls_solve_system<doxid-galahad__uls_8h_1a01d3e7c19415125c660eba51d99c7518>`(
		void** data,
		int* status,
		int m,
		int n,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` sol[],
		bool trans
	);

	void :ref:`uls_information<doxid-galahad__uls_8h_1ab41cc4ccba208d7de3a0ccbc4b4efbcf>`(void** data, struct :ref:`uls_inform_type<doxid-structuls__inform__type>`* inform, int* status);

	void :ref:`uls_terminate<doxid-galahad__uls_8h_1a36b2ea1ade2cdd8bca238f46e9e98435>`(
		void** data,
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>`* inform
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

.. index:: pair: function; uls_initialize
.. _doxid-galahad__uls_8h_1a7afb5f2dde112e60686a5527a8f37ca4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_initialize(
		const char solver[],
		void** data,
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		int* status
	)

Set default control values and initialize private data

Select solver, set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- solver

		- is a one-dimensional array of type char that specifies the :ref:`solver package <doxid->` that should be used to factorize the matrix :math:`A`. It should be one of 'gls', 'ma28', 'ma48 or 'getr'; lower or upper case variants are allowed.

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful.
		  
		  * -26. The requested solver is not available.

.. index:: pair: function; uls_read_specfile
.. _doxid-galahad__uls_8h_1a5e2c9573bc8661114e9f073782b460ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_read_specfile(struct :ref:`uls_control_type<doxid-structuls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNULS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/uls.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; uls_factorize_matrix
.. _doxid-galahad__uls_8h_1a6c0599479b84ee7d7c4ee7c473b76a83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_factorize_matrix(
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char type[],
		int ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` val[],
		const int row[],
		const int col[],
		const int ptr[]
	)

Import matrix data into internal storage prior to solution, analyse the sparsity patern, and subsequently factorize the matrix



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The import, analysis and factorization were conducted succesfully.
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m> 0 or requirement that the matrix type must contain the relevant string 'dense', 'coordinate' or 'sparse_by_rows has been violated.
		  
		  * -26. The requested solver is not available.
		  
		  * -29. This option is not available with this solver.
		  
		  * -32. More than control.max integer factor size words of internal integer storage are required for in-core factorization.
		  
		  * -50. A solver-specific error occurred; check the solver-specific information component of inform along with the solver’s documentation for more details.

	*
		- m

		- is a scalar variable of type int, that holds the number of rows in the unsymmetric matrix :math:`A`.

	*
		- n

		- is a scalar variable of type int, that holds the number of columns in the unsymmetric matrix :math:`A`.

	*
		- type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the matrix :math:`A`. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type int, that holds the number of entries in :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- val

		- is a one-dimensional array of size ne and type double, that holds the values of the entries of the matrix :math:`A` in any of the supported storage schemes.

	*
		- row

		- is a one-dimensional array of size ne and type int, that holds the row indices of the matrix :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- col

		- is a one-dimensional array of size ne and type int, that holds the column indices of the matrix :math:`A` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense storage schemes is used, and in this case can be NULL.

	*
		- ptr

		- is a one-dimensional array of size m+1 and type int, that holds the starting position of each row of the matrix :math:`A`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; uls_reset_control
.. _doxid-galahad__uls_8h_1ad2ad6daa4d54d75e40fbe253f2bc5881:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_reset_control(
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful.

.. index:: pair: function; uls_solve_system
.. _doxid-galahad__uls_8h_1a01d3e7c19415125c660eba51d99c7518:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_solve_system(
		void** data,
		int* status,
		int m,
		int n,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` sol[],
		bool trans
	)

Solve the linear system :math:`Ax=b` or :math:`A^Tx=b`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The required solution was obtained.
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -34. The package PARDISO failed; check the solver-specific information components inform.pardiso iparm and inform.pardiso_dparm along with PARDISO’s documentation for more details.
		  
		  * -35. The package WSMP failed; check the solver-specific information components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s documentation for more details.

	*
		- m

		- is a scalar variable of type int, that holds the number of rows in the unsymmetric matrix :math:`A`.

	*
		- n

		- is a scalar variable of type int, that holds the number of columns in the unsymmetric matrix :math:`A`.

	*
		- sol

		- is a one-dimensional array of size n and type double. On entry, it must hold the vector :math:`b`. On a successful exit, its contains the solution :math:`x`.

	*
		- trans

		- is a scalar variable of type bool, that specifies whether to solve the equation :math:`A^Tx=b` (trans=true) or :math:`Ax=b` (trans=false).

.. index:: pair: function; uls_information
.. _doxid-galahad__uls_8h_1ab41cc4ccba208d7de3a0ccbc4b4efbcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_information(void** data, struct :ref:`uls_inform_type<doxid-structuls__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`uls_inform_type <doxid-structuls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; uls_terminate
.. _doxid-galahad__uls_8h_1a36b2ea1ade2cdd8bca238f46e9e98435:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void uls_terminate(
		void** data,
		struct :ref:`uls_control_type<doxid-structuls__control__type>`* control,
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`uls_inform_type <doxid-structuls__inform__type>`)

