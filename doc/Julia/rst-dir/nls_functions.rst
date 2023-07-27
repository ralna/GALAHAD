.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_nls_subproblem_control_type.rst
	struct_nls_control_type.rst
	struct_nls_subproblem_inform_type.rst
	struct_nls_inform_type.rst
	struct_nls_time_type.rst

function calls
--------------

.. index:: pair: function; nls_initialize
.. _doxid-galahad__nls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_initialize(
		void** data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

.. index:: pair: function; nls_read_specfile
.. _doxid-galahad__nls_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_read_specfile(struct :ref:`nls_control_type<doxid-structnls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNNLS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/nls.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; nls_import
.. _doxid-galahad__nls_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_import(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void** data,
		int* status,
		int n,
		int m,
		const char J_type[],
		int J_ne,
		const int J_row[],
		const int J_col[],
		const int J_ptr[],
		const char H_type[],
		int H_ne,
		const int H_row[],
		const int H_col[],
		const int H_ptr[],
		const char P_type[],
		int P_ne,
		const int P_row[],
		const int P_col[],
		const int P_ptr[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` w[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

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

		  * -3. The restrictions n > 0, m > 0 or requirement that J/H/P_type contains its relevant string 'dense', 'dense_by_columns', 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'diagonal' or 'absent' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- J_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the Jacobian, :math:`J`. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- J_ne

		- is a scalar variable of type int, that holds the number of entries in :math:`J` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- J_row

		- is a one-dimensional array of size J_ne and type int, that holds the row indices of :math:`J` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- J_col

		- is a one-dimensional array of size J_ne and type int, that holds the column indices of :math:`J` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- J_ptr

		- is a one-dimensional array of size m+1 and type int, that holds the starting position of each row of :math:`J`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, :math:`H`. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to :math:`H` is via matrix-vector products; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type int, that holds the row indices of the lower triangular part of :math:`H` in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type int, that holds the column indices of the lower triangular part of :math:`H` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of :math:`H`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- P_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the residual-Hessians-vector product matrix, :math:`P`. It should be one of 'coordinate', 'sparse_by_columns', 'dense_by_columns' or 'absent', the latter if access to :math:`P` is via matrix-vector products; lower or upper case variants are allowed.

	*
		- P_ne

		- is a scalar variable of type int, that holds the number of entries in :math:`P` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- P_row

		- is a one-dimensional array of size P_ne and type int, that holds the row indices of :math:`P` in either the sparse co-ordinate, or the sparse column-wise storage scheme. It need not be set when the dense storage scheme is used, and in this case can be NULL.

	*
		- P_col

		- is a one-dimensional array of size P_ne and type int, that holds the row indices of :math:`P` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- P_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of :math:`P`, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- w

		- is a one-dimensional array of size m and type double, that holds the values :math:`w` of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

.. index:: pair: function; nls_reset_control
.. _doxid-galahad__nls_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_reset_control(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; nls_solve_with_mat
.. _doxid-galahad__nls_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_solve_with_mat(
		void** data,
		void* userdata,
		int* status,
		int n,
		int m,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		int(*)(int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const void*) eval_c,
		int j_ne,
		int(*)(int, int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const void*) eval_j,
		int h_ne,
		int(*)(int, int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const void*) eval_h,
		int p_ne,
		int(*)(int, int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], bool, const void*) eval_hprods
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where :math:`H = \nabla_{xx}f(x)` is provided specifically, and all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs.

	*
		- status

		-
		  is a scalar variable of type int, that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the residual :math:`c(x)`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- eval_c

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_c( int n, const double x[], double c[], const void *userdata )

		  The componnts of the residual function :math:`c(x)` evaluated at x= :math:`x` must be assigned to c, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_c`` via the structure ``userdata``.

	*
		- j_ne

		- is a scalar variable of type int, that holds the number of entries in the Jacobian matrix :math:`J`.

	*
		- eval_j

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_j( int n, int m, int jne, const double x[], double j[],
		  	            const void *userdata )

		  The components of the Jacobian :math:`J = \nabla_x c(x`) of the residuals must be assigned to j in the same order as presented to nls_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_j`` via the structure ``userdata``.

	*
		- h_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H` if it is used.

	*
		- eval_h

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_h( int n, int m, int hne, const double x[], const double y[],
		  	            double h[], const void *userdata )

		  The nonzeros of the matrix :math:`H = \sum_{i=1}^m y_i \nabla_{xx}c_i(x)` of the weighted residual Hessian evaluated at x= :math:`x` and y= :math:`y` must be assigned to h in the same order as presented to nls_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_h`` via the structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type int, that holds the number of entries in the residual-Hessians-vector product matrix :math:`P` if it is used.

	*
		- eval_hprods

		-
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_hprods( int n, int m, int pne, const double x[],
		  	                    const double v[], double p[], bool got_h,
		  	                    const void *userdata ) );

		  The entries of the matrix :math:`P`, whose i-th column is the product :math:`\nabla_{xx}c_i(x) v` between :math:`\nabla_{xx}c_i(x)`, the Hessian of the i-th component of the residual :math:`c(x)` at x= :math:`x`, and v= :math:`v` must be returned in p and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_without_mat
.. _doxid-galahad__nls_8h_1a692ecbfaa428584e60aa4c33d7278a64:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_solve_without_mat(
		void** data,
		void* userdata,
		int* status,
		int n,
		int m,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		int(*)(int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const void*) eval_c,
		int(*)(int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const bool, :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], bool, const void*) eval_jprod,
		int(*)(int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], bool, const void*) eval_hprod,
		int p_ne,
		int(*)(int, int, int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], bool, const void*) eval_hprods
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to :math:`H = \nabla_{xx}f(x)` is provided by Hessian-vector products, and all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs.

	*
		- status

		-
		  is a scalar variable of type int, that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the residual :math:`c(x)`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- eval_c

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_c( int n, const double x[], double c[], const void *userdata )

		  The componnts of the residual function :math:`c(x)` evaluated at x= :math:`x` must be assigned to c, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_c`` via the structure ``userdata``.

	*
		- eval_jprod

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_jprod( int n, int m, const double x[], bool transpose,
		  	                double u[], const double v[], bool got_j,
		  	                const void *userdata )

		  The sum :math:`u + \nabla_{x}c_(x) v` (if tranpose is false) or The sum :math:`u + (\nabla_{x}c_(x))^T v` (if tranpose is true) bewteen the product of the Jacobian :math:`\nabla_{x}c_(x)` or its tranpose with the vector v= :math:`v` and the vector $ :math:`u` must be returned in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jprod`` via the structure ``userdata``.

	*
		- eval_hprod

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_hprod( int n, int m, const double x[], const double y[],
		  	                double u[], const double v[], bool got_h,
		  	                const void *userdata )

		  The sum :math:`u + \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v` of the product of the weighted residual Hessian :math:`H = \sum_{i=1}^m y_i \nabla_{xx}c_i(x)` evaluated at x= :math:`x` and y= :math:`y` with the vector v= :math:`v` and the vector $ :math:`u` must be returned in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. The Hessians have already been evaluated or used at x if got_h is true. Data may be passed into ``eval_hprod`` via the structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type int, that holds the number of entries in the residual-Hessians-vector product matrix :math:`P` if it is used.

	*
		- eval_hprods

		-
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_hprods( int n, int m, int p_ne, const double x[],
		  	                 const double v[], double pval[], bool got_h,
		  	                 const void *userdata )

		  The entries of the matrix :math:`P`, whose i-th column is the product :math:`\nabla_{xx}c_i(x) v` between :math:`\nabla_{xx}c_i(x)`, the Hessian of the i-th component of the residual :math:`c(x)` at x= :math:`x`, and v= :math:`v` must be returned in pval and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_reverse_with_mat
.. _doxid-galahad__nls_8h_1a9ad89605640c53c33ddd5894b5e3edd1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_solve_reverse_with_mat(
		void** data,
		int* status,
		int* eval_status,
		int n,
		int m,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		int j_ne,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` J_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		int h_ne,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` H_val[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[],
		int p_ne,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` P_val[]
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where :math:`H = \nabla_{xx}f(x)` is provided specifically, but function/derivative information is only available by returning to the calling procedure



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

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.



		  * 2. The user should compute the vector of residuals :math:`c(x)` at the point :math:`x` indicated in x and then re-enter the function. The required value should be set in c, and eval_status should be set to 0. If the user is unable to evaluate :math:`c(x)` for instance, if the function is undefined at :math:`x` the user need not set c, but should then set eval_status to a non-zero value.



		  * 3. The user should compute the Jacobian of the vector of residual functions, :math:`\nabla_x c(x)`, at the point :math:`x` indicated in x and then re-enter the function. The l-th component of the Jacobian stored according to the scheme specified for the remainder of :math:`J` in the earlier call to nls_import should be set in J_val[l], for l = 0, ..., J_ne-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`J` for instance, if a component of the matrix is undefined at :math:`x` the user need not set J_val, but should then set eval_status to a non-zero value.

		  * 4. The user should compute the matrix :math:`H = \sum_{i=1}^m v_i \nabla_{xx}c_i(x)` of weighted residual Hessian evaluated at x= :math:`x` and v= :math:`v` and then re-enter the function. The l-th component of the matrix stored according to the scheme specified for the remainder of :math:`H` in the earlier call to nls_import should be set in H_val[l], for l = 0, ..., H_ne-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`H` for instance, if a component of the matrix is undefined at :math:`x` the user need not set H_val, but should then set eval_status to a non-zero value. **Note** that this return will not happen if the Gauss-Newton model is selected.

		  * 7. The user should compute the entries of the matrix :math:`P`, whose i-th column is the product :math:`\nabla_{xx}c_i(x) v` between :math:`\nabla_{xx}c_i(x)`, the Hessian of the i-th component of the residual :math:`c(x)` at x= :math:`x`, and v= :math:`v` and then re-enter the function. The l-th component of the matrix stored according to the scheme specified for the remainder of :math:`P` in the earlier call to nls_import should be set in P_val[l], for l = 0, ..., P_ne-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`P` for instance, if a component of the matrix is undefined at :math:`x` the user need not set P_val, but should then set eval_status to a non-zero value. **Note** that this return will not happen if either the Gauss-Newton or Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type int, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the residual :math:`c(x)`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- j_ne

		- is a scalar variable of type int, that holds the number of entries in the Jacobian matrix :math:`J`.

	*
		- J_val

		- is a one-dimensional array of size j_ne and type double, that holds the values of the entries of the Jacobian matrix :math:`J` in any of the available storage schemes. See status = 3, above, for more details.

	*
		- y

		- is a one-dimensional array of size m and type double, that is used for reverse communication. See status = 4 above for more details.

	*
		- h_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes. See status = 4, above, for more details.

	*
		- v

		- is a one-dimensional array of size n and type double, that is used for reverse communication. See status = 7, above, for more details.

	*
		- p_ne

		- is a scalar variable of type int, that holds the number of entries in the residual-Hessians-vector product matrix, :math:`P`.

	*
		- P_val

		- is a one-dimensional array of size p_ne and type double, that holds the values of the entries of the residual-Hessians-vector product matrix, :math:`P`. See status = 7, above, for more details.

.. index:: pair: function; nls_solve_reverse_without_mat
.. _doxid-galahad__nls_8h_1a6dddd928c19adec0abf76bdb2d75da17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_solve_reverse_without_mat(
		void** data,
		int* status,
		int* eval_status,
		int n,
		int m,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		bool* transpose,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		int p_ne,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` P_val[]
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to :math:`H = \nabla_{xx}f(x)` is provided by Hessian-vector products, but function/derivative information is only available by returning to the calling procedure.



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

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.



		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.



		  * 2. The user should compute the vector of residuals :math:`c(x)` at the point :math:`x` indicated in x and then re-enter the function. The required value should be set in c, and eval_status should be set to 0. If the user is unable to evaluate :math:`c(x)` for instance, if the function is undefined at :math:`x` the user need not set c, but should then set eval_status to a non-zero value.



		  * 5. The user should compute the sum :math:`u + \nabla_{x}c_(x) v` (if tranpose is false) or :math:`u + (\nabla_{x}c_(x))^T v` (if tranpose is true) between the product of the Jacobian :math:`\nabla_{x}c_(x)` or its tranpose with the vector v= :math:`v` and the vector u = $ :math:`u`, and then re-enter the function. The result should be set in u, and eval_status should be set to 0. If the user is unable to evaluate the sum for instance, if the Jacobian is undefined at :math:`x` the user need not set u, but should then set eval_status to a non-zero value.



		  * 6. The user should compute the sum :math:`u + \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v` between the product of the weighted residual Hessian :math:`H = \sum_{i=1}^m y_i \nabla_{xx}c_i(x)` evaluated at x= :math:`x` and y= :math:`y` with the vector v= :math:`v` and the the vector u = $ :math:`u`, and then re-enter the function. The result should be set in u, and eval_status should be set to 0. If the user is unable to evaluate the sum for instance, if the weifghted residual Hessian is undefined at :math:`x` the user need not set u, but should then set eval_status to a non-zero value.



		  * 7. The user should compute the entries of the matrix :math:`P`, whose i-th column is the product :math:`\nabla_{xx}c_i(x) v` between :math:`\nabla_{xx}c_i(x)`, the Hessian of the i-th component of the residual :math:`c(x)` at x= :math:`x`, and v= :math:`v` and then re-enter the function. The l-th component of the matrix stored according to the scheme specified for the remainder of :math:`P` in the earlier call to nls_import should be set in P_val[l], for l = 0, ..., P_ne-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`P` for instance, if a component of the matrix is undefined at :math:`x` the user need not set P_val, but should then set eval_status to a non-zero value. **Note** that this return will not happen if either the Gauss-Newton or Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type int, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the residual :math:`c(x)`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- transpose

		- is a scalar variable of type bool, that indicates whether the product with Jacobian or its transpose should be obtained when status=5.

	*
		- u

		- is a one-dimensional array of size max(n,m) and type double, that is used for reverse communication. See status = 5,6 above for more details.

	*
		- v

		- is a one-dimensional array of size max(n,m) and type double, that is used for reverse communication. See status = 5,6,7 above for more details.

	*
		- y

		- is a one-dimensional array of size m and type double, that is used for reverse communication. See status = 6 above for more details.

	*
		- p_ne

		- is a scalar variable of type int, that holds the number of entries in the residual-Hessians-vector product matrix, :math:`P`.

	*
		- P_val

		- is a one-dimensional array of size P_ne and type double, that holds the values of the entries of the residual-Hessians-vector product matrix, :math:`P`. See status = 7, above, for more details.

.. index:: pair: function; nls_information
.. _doxid-galahad__nls_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_information(void** data, struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; nls_terminate
.. _doxid-galahad__nls_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void nls_terminate(
		void** data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)
