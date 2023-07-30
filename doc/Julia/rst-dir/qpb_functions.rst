.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_qpb_control_type.rst
	struct_qpb_inform_type.rst
	struct_qpb_time_type.rst

function calls
--------------

.. index:: pair: function; qpb_initialize
.. _doxid-galahad__qpb_8h_1afdfb45debf15ddcc3d6e0feea4e34784:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_initialize(void** data, structure :ref:`qpb_control_type<doxid-structqpb__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`qpb_control_type <doxid-structqpb__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; qpb_read_specfile
.. _doxid-galahad__qpb_8h_1a162514dc80468d390e80c620890c8710:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_read_specfile(struct :ref:`qpb_control_type<doxid-structqpb__control__type>`* control, const Vararg{Cchar} specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNQPB.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/qpb.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`qpb_control_type <doxid-structqpb__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; qpb_import
.. _doxid-galahad__qpb_8h_1a36559bb33d97b61e94e0c7dfa73b67d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_import(
		struct :ref:`qpb_control_type<doxid-structqpb__control__type>`* control,
		void** data,
		int* status,
		Int32 n,
		Int32 m,
		const Vararg{Cchar} H_type[],
		Int32 H_ne,
		const int H_row[],
		const int H_col[],
		const int H_ptr[],
		const Vararg{Cchar} A_type[],
		Int32 A_ne,
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

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`qpb_control_type <doxid-structqpb__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 0. The import was succesful

		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restrictions n > 0 or m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none' has been violated.

		  * -23. An entry from the strict upper triangle of :math:`H` has been specified.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables.

	*
		- m

		- is a scalar variable of type Int32 that holds the number of general linear constraints.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, :math:`H`. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if :math:`H=0`; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type Int32 that holds the row indices of the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type Int32 that holds the column indices of the lower triangular part of :math:`H` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type Int32 that holds the starting position of each row of the lower triangular part of :math:`H`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, :math:`A`. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type Int32 that holds the number of entries in :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type Int32 that holds the row indices of :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type Int32 that holds the column indices of :math:`A` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type Int32 that holds the starting position of each row of :math:`A`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; qpb_reset_control
.. _doxid-galahad__qpb_8h_1a42484e4b1aafb880fcdb215d5683a652:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_reset_control(
		struct :ref:`qpb_control_type<doxid-structqpb__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`qpb_control_type <doxid-structqpb__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 0. The import was succesful.

.. index:: pair: function; qpb_solve_qp
.. _doxid-galahad__qpb_8h_1ac83f8c83e64a7f130fb247f162169472:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_solve_qp(
		void** data,
		int* status,
		Int32 n,
		Int32 m,
		Int32 h_ne,
		const T H_val[],
		const T g[],
		const T f,
		Int32 a_ne,
		const T A_val[],
		const T c_l[],
		const T c_u[],
		const T x_l[],
		const T x_u[],
		T x[],
		T c[],
		T y[],
		T z[],
		Int32 x_stat[],
		Int32 c_stat[]
	)

Solve the quadratic program when the Hessian :math:`H` is available.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  Possible exit are:

		  * 0. The run was succesful.



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restrictions n > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none' has been violated.

		  * -5. The simple-bound constraints are inconsistent.

		  * -7. The constraints appear to have no feasible point.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -23. An entry from the strict upper triangle of :math:`H` has been specified.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- m

		- is a scalar variable of type Int32 that holds the number of general linear constraints.

	*
		- h_ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term :math:`g` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- f

		- is a scalar of type T that holds the constant term :math:`f` of the objective function.

	*
		- a_ne

		- is a scalar variable of type Int32 that holds the number of entries in the constraint Jacobian matrix :math:`A`.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type T that holds the values of the entries of the constraint Jacobian matrix :math:`A` in any of the available storage schemes.

	*
		- c_l

		- is a one-dimensional array of size m and type T that holds the lower bounds :math:`c^l` on the constraints :math:`A x`. The i-th component of c_l, i = 0, ... , m-1, contains :math:`c^l_i`.

	*
		- c_u

		- is a one-dimensional array of size m and type T that holds the upper bounds :math:`c^l` on the constraints :math:`A x`. The i-th component of c_u, i = 0, ... , m-1, contains :math:`c^u_i`.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the lower bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_l, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the upper bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_u, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual :math:`c(x)`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values :math:`y` of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , n-1, contains :math:`y_j`.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type Int32 that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type Int32 that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value :math:`a_i^Tx` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; qpb_information
.. _doxid-galahad__qpb_8h_1a2b77d21a6c613462657e8b7f51b6b1d2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_information(void** data, structure :ref:`qpb_inform_type<doxid-structqpb__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`qpb_inform_type <doxid-structqpb__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; qpb_terminate
.. _doxid-galahad__qpb_8h_1a35596f23213b063f1f8abfc6f796bc77:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void qpb_terminate(
		void** data,
		struct :ref:`qpb_control_type<doxid-structqpb__control__type>`* control,
		struct :ref:`qpb_inform_type<doxid-structqpb__inform__type>`* inform
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

		- is a structure containing control information (see :ref:`qpb_control_type <doxid-structqpb__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`qpb_inform_type <doxid-structqpb__inform__type>`)
