.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_trs_control_type.rst
	struct_trs_inform_type.rst
	struct_trs_time_type.rst

.. _details-global:

function calls
--------------

.. ---------------------------------------------------------------------------
.. _global:

Global Scope
============


.. toctree::
	:hidden:

	namespace_conf.rst
	struct_trs_control_type.rst
	struct_trs_history_type.rst
	struct_trs_inform_type.rst
	struct_trs_time_type.rst

Overview
~~~~~~~~



.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	-- modules

	:ref:`conf<doxid-namespaceconf>`

	-- table types

	:ref:`trs_control_type<doxid-structtrs__control__type>`
	:ref:`trs_history_type<doxid-structtrs__history__type>`
	:ref:`trs_inform_type<doxid-structtrs__inform__type>`
	:ref:`trs_time_type<doxid-structtrs__time__type>`

	-- functions

	function :ref:`trs_initialize<doxid-galahad__trs_8h_1acb066d992c4ec394402bc7b7317e1163>`(data, control, status)
	function :ref:`trs_read_specfile<doxid-galahad__trs_8h_1adc7c56e7be2f7cc9d32921582d379b13>`(control, specfile)
	function :ref:`trs_import<doxid-galahad__trs_8h_1a4becded30e9b95fe7028b7799292c0af>`(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
	function :ref:`trs_import_m<doxid-galahad__trs_8h_1a427420b6025d522bb7b3c652e8c2be48>`(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
	function :ref:`trs_import_a<doxid-galahad__trs_8h_1ad726ff8f6c25c4384d2b952e8fab4409>`(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
	function :ref:`trs_reset_control<doxid-galahad__trs_8h_1aae677e64bacb35354f49326815b694c3>`(control, data, status)

	function :ref:`trs_solve_problem<doxid-galahad__trs_8h_1aadb8a751c29efcef663bf9560a1f9a8e>`(
		data,
		status,
		n,
		radius,
		f,
		c,
		H_ne,
		H_val,
		x,
		M_ne,
		M_val,
		m,
		A_ne,
		A_val,
		y
		)

	function :ref:`trs_information<doxid-galahad__trs_8h_1a3dda24010e564e2d6536cc7ea518451e>`(data, inform, status)
	function :ref:`trs_terminate<doxid-galahad__trs_8h_1ab5cf0077db0631814fdd03599a585376>`(data, control, inform)

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Functions
---------

.. index:: pair: function; trs_initialize
.. _doxid-galahad__trs_8h_1acb066d992c4ec394402bc7b7317e1163:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_initialize(data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`trs_control_type <doxid-structtrs__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; trs_read_specfile
.. _doxid-galahad__trs_8h_1adc7c56e7be2f7cc9d32921582d379b13:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNTRS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/trs.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`trs_control_type <doxid-structtrs__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; trs_import
.. _doxid-galahad__trs_8h_1a4becded30e9b95fe7028b7799292c0af:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_import(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`trs_control_type <doxid-structtrs__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', diagonal' or 'identity' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of rows (and columns) of H.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, :math:`H`. It should be one of 'coordinate', 'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type int, that holds the row indices of the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type int, that holds the column indices of the lower triangular part of :math:`H` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of :math:`H`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; trs_import_m
.. _doxid-galahad__trs_8h_1a427420b6025d522bb7b3c652e8c2be48:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_import_m(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)

Import data for the scaling matrix M into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', diagonal' or 'identity' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of rows (and columns) of M.

	*
		- M_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the scaling matrix, :math:`M`. It should be one of 'coordinate', 'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants are allowed.

	*
		- M_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of :math:`M` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- M_row

		- is a one-dimensional array of size M_ne and type int, that holds the row indices of the lower triangular part of :math:`M` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- M_col

		- is a one-dimensional array of size M_ne and type int, that holds the column indices of the lower triangular part of :math:`M` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or identity storage schemes are used, and in this case can be NULL.

	*
		- M_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of :math:`M`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; trs_import_a
.. _doxid-galahad__trs_8h_1ad726ff8f6c25c4384d2b952e8fab4409:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_import_a(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)

Import data for the constraint matrix A into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.

	*
		- m

		- is a scalar variable of type int, that holds the number of general linear constraints, i.e., the number of rows of A, if any. m must be non-negative.

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

.. index:: pair: function; trs_reset_control
.. _doxid-galahad__trs_8h_1aae677e64bacb35354f49326815b694c3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_reset_control(control, data, status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`trs_control_type <doxid-structtrs__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was succesful.

.. index:: pair: function; trs_solve_problem
.. _doxid-galahad__trs_8h_1aadb8a751c29efcef663bf9560a1f9a8e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_solve_problem(
		data,
		status,
		n,
		radius,
		f,
		c,
		H_ne,
		H_val,
		x,
		M_ne,
		M_val,
		m,
		A_ne,
		A_val,
		y
		)

Solve the trust-region problem.



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
		  
		  On initial entry, status must be set to 1.
		  
		  Possible exit are:
		  
		  * 0. The run was succesful.
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0, radius > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'identity' has been violated.
		  
		  * -9. The analysis phase of the factorization of the matrix (1) failed.
		  
		  * -10. The factorization of the matrix (1) failed.
		  
		  * -15. The matrix M appears not to be diagonally dominant.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.
		  
		  * -18. Too many factorizations have been required. This may happen if control.max factorizations is too small, but may also be symptomatic of a badly scaled problem.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- radius

		- is a scalar of type double, that holds the trust-region radius, :math:`\Delta`, used. radius must be strictly positive

	*
		- f

		- is a scalar of type double, that holds the constant term :math:`f` of the objective function.

	*
		- c

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`c` of the objective function. The j-th component of c, j = 0, ... , n-1, contains :math:`c_j`.

	*
		- H_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- M_ne

		- is a scalar variable of type int, that holds the number of entries in the scaling matrix :math:`M` if it not the identity matrix.

	*
		- M_val

		- is a one-dimensional array of size M_ne and type double, that holds the values of the entries of the scaling matrix :math:`M`, if it is not the identity matrix, in any of the available storage schemes. If M_val is NULL, M will be taken to be the identity matrix.

	*
		- m

		- is a scalar variable of type int, that holds the number of general linear constraints, if any. m must be non-negative.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in the constraint Jacobian matrix :math:`A` if used. A_ne must be non-negative.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type double, that holds the values of the entries of the constraint Jacobian matrix :math:`A`, if used, in any of the available storage schemes. If A_val is NULL, no constraints will be enforced.

	*
		- y

		- is a one-dimensional array of size m and type double, that holds the values :math:`y` of the Lagrange multipliers for the equality constraints :math:`A x = 0` if used. The i-th component of y, i = 0, ... , m-1, contains :math:`y_i`.

.. index:: pair: function; trs_information
.. _doxid-galahad__trs_8h_1a3dda24010e564e2d6536cc7ea518451e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_information(data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`trs_inform_type <doxid-structtrs__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; trs_terminate
.. _doxid-galahad__trs_8h_1ab5cf0077db0631814fdd03599a585376:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function trs_terminate(data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`trs_control_type <doxid-structtrs__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`trs_inform_type <doxid-structtrs__inform__type>`)

