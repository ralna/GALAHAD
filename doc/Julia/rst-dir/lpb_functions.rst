.. _global:

callable functions
------------------

.. index:: pair: function; lpb_initialize
.. _doxid-galahad__lpb_8h_1a63dd5d968d870274e0abc9c3e1e553f6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_initialize(data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lpb_control_type <doxid-structlpb__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; lpb_read_specfile
.. _doxid-galahad__lpb_8h_1ad3cf4da5c65e4b31d2d4ff45a392c567:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNLPB.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/lpb.pdf for a list of keywords that may be set.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lpb_control_type <doxid-structlpb__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; lpb_import
.. _doxid-galahad__lpb_8h_1ac3308e860ab39acf9d7f293f75d80fbd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_import(control, data, status, n, m, 
                            A_type, A_ne, A_row, A_col, A_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`lpb_control_type <doxid-structlpb__control__type>`)

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

		  * -3. The restrictions n > 0 or m > 0 or requirement that A_type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables.

	*
		- m

		- is a scalar variable of type Int32 that holds the number of general linear constraints.

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

.. index:: pair: function; lpb_reset_control
.. _doxid-galahad__lpb_8h_1aac79b2577895e28d4a92deb9f3bd24a6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_reset_control(control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`lpb_control_type <doxid-structlpb__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 0. The import was succesful.

.. index:: pair: function; lpb_solve_lp
.. _doxid-galahad__lpb_8h_1a3fecba0e7ec404089d904a5623e0e83e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_solve_lp(data, status, n, m, g, f, a_ne, A_val, 
                              c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)

Solve the linear program.

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

		  * -3. The restrictions n > 0 and m > 0 or requirement that A_type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.

		  * -5. The simple-bound constraints are inconsistent.

		  * -7. The constraints appear to have no feasible point.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- m

		- is a scalar variable of type Int32 that holds the number of general linear constraints.

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

		- is a one-dimensional array of size m and type T that holds the residual :math:`c(x)`. The i-th component of c, i = 0, ... , m-1, contains :math:`c_i(x)`.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values :math:`y` of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , m-1, contains :math:`y_i`.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type Int32 that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type Int32 that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value :math:`a_i^T x` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; lpb_information
.. _doxid-galahad__lpb_8h_1ad3b3173cbeb7a9b01995d678324cbe4e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_information(data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lpb_inform_type <doxid-structlpb__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; lpb_terminate
.. _doxid-galahad__lpb_8h_1ac7258f2afb0b15c191838ecfa377d264:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lpb_terminate(data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lpb_control_type <doxid-structlpb__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lpb_inform_type <doxid-structlpb__inform__type>`)
