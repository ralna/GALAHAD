.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_blls_control_type.rst
	struct_blls_inform_type.rst
	struct_blls_time_type.rst

function calls
--------------

.. index:: pair: function; blls_initialize
.. _doxid-galahad__blls_8h_1a12708c98f2473e03cd46f4dcfdb03409:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_initialize(
		void** data,
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
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

		- is a struct containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; blls_read_specfile
.. _doxid-galahad__blls_8h_1aa24c9c2fdaaaac84df5b98abbf84c859:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_read_specfile(
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNBLLS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/blls.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; blls_import
.. _doxid-galahad__blls_8h_1afacd84f0b7592f4532cf7b77d278282f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_import(
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
		void** data,
		int* status,
		int n,
		int m,
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

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

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

		  * -3. The restrictions n > 0, m > 0 or requirement that type contains its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns'; has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the Jacobian :math:`A`. It should be one of 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns'; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in :math:`A` in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type int, that holds the row indices of :math:`A` in the sparse co-ordinate or sparse column-wise storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of :math:`A` in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 or m+1 and type int, that holds the starting position of each row of :math:`A`, as well as the total number of entries, in the sparse row-wise storage scheme, or the starting position of each column of :math:`A`, as well as the total number of entries, in the sparse column-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; blls_import_without_a
.. _doxid-galahad__blls_8h_1a419f9b0769b4389beffbbc5f7d0fd58c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_import_without_a(
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
		void** data,
		int* status,
		int n,
		int m
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

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

		  * -3. The restriction n > 0 or m > 0 has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

.. index:: pair: function; blls_reset_control
.. _doxid-galahad__blls_8h_1a96981ac9a0e3f44b2b38362fc3ab9991:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_reset_control(
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; blls_solve_given_a
.. _doxid-galahad__blls_8h_1acf6d292989a5ac09f7f3e507283fb5bf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_solve_given_a(
		void** data,
		void* userdata,
		int* status,
		int n,
		int m,
		int A_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		int x_stat[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` w[],
		int(*)(int, const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`[], const void*) eval_prec
	)

Solve the bound-constrained linear least-squares problem when the Jacobian :math:`A` is available.



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

		  * 0. The run was succesful.



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restrictions n > 0, m > 0 or requirement that a type contains its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense_by_rows' or 'dense_by_columns' has been violated.

		  * -4. The simple-bound constraints are inconsistent.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the constant term :math:`b` in the residuals. The i-th component of b, i = 0, ... , m-1, contains :math:`b_i`.

	*
		- x_l

		- is a one-dimensional array of size n and type double, that holds the lower bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_l, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x_u

		- is a one-dimensional array of size n and type double, that holds the upper bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_u, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- z

		- is a one-dimensional array of size n and type double, that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the values of the residuals :math:`c = A x - b`. The i-th component of c, i = 0, ... , m-1, contains :math:`c_i`.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the values of the gradient :math:`g = A^T c`. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type int, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- w

		- is an optional one-dimensional array of size m and type double, that holds the values :math:`w` of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

	*
		- eval_prec

		-
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:

		  .. ref-code-block:: julia

		  	int eval_prec( int n, const double v[], double p[],
		  	               const void *userdata )

		  The product :math:`p = P^{-1} v` involving the user's preconditioner :math:`P` with the vector v = :math:`v`, the result :math:`p` must be retured in p, and the function return value set to 0. If the evaluation is impossible, return should be set to a nonzero value. Data may be passed into ``eval_prec`` via the structure ``userdata``.

.. index:: pair: function; blls_solve_reverse_a_prod
.. _doxid-galahad__blls_8h_1ac139bc1c65cf12cb532c4ab09f3af9d0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_solve_reverse_a_prod(
		void** data,
		int* status,
		int* eval_status,
		int n,
		int m,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		int x_stat[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` p[],
		int nz_v[],
		int* nz_v_start,
		int* nz_v_end,
		const int nz_p[],
		int nz_p_end,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` w[]
	)

Solve the bound-constrained linear least-squares problem when the products of the Jacobian :math:`A` and its transpose with specified vectors may be computed by the calling program.



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

		  * -3. The restriction n > 0 or requirement that a type contains its relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense_by_rows' or 'dense_by_columns' has been violated.

		  * -4. The simple-bound constraints are inconsistent.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * 2. The product :math:`Av` of the residual Jacobian :math:`A` with a given output vector :math:`v` is required from the user. The vector :math:`v` will be stored in v and the product :math:`Av` must be returned in p, status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged. If the product cannot be formed, v need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value.



		  * 3. The product :math:`A^Tv` of the transpose of the residual Jacobian :math:`A` with a given output vector :math:`v` is required from the user. The vector :math:`v` will be stored in v and the product :math:`A^Tv` must be returned in p, status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged. If the product cannot be formed, v need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value.



		  * 4. The product :math:`Av` of the residual Jacobian :math:`A` with a given sparse output vector :math:`v` is required from the user. The nonzero components of the vector :math:`v` will be stored as entries nz_in[nz_in_start-1:nz_in_end-1] of v and the product :math:`Av` must be returned in p, status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged; The remaining components of v should be ignored. If the product cannot be formed, v need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value.



		  * 5. The nonzero components of the product :math:`Av` of the residual Jacobian :math:`A` with a given sparse output vector :math:`v` is required from the user. The nonzero components of the vector :math:`v` will be stored as entries nz_in[nz_in_start-1:nz_in_end-1] of v; the remaining components of v should be ignored. The resulting **nonzeros** in the product :math:`Av` must be placed in their appropriate comnponents of p, while a list of indices of the nonzeros placed in nz_out[0 : nz_out_end-1] and the number of nonzeros recorded in nz_out_end. Additionally, status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged. If the product cannot be formed, v, nz_out_end and nz_out need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value.



		  * 6. A subset of the product :math:`A^Tv` of the transpose of the residual Jacobian :math:`A` with a given output vector :math:`v` is required from the user. The vector :math:`v` will be stored in v and components nz_in[nz_in_start-1:nz_in_end-1] of the product :math:`A^Tv` must be returned in the relevant components of p (the remaining components should not be set), status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged. If the product cannot be formed, v need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value.



		  * 7. The product :math:`P^{-1}v` of the inverse of the preconditioner :math:`P` with a given output vector :math:`v` is required from the user. The vector :math:`v` will be stored in v and the product :math:`P^{-1} v` must be returned in p, status_eval should be set to 0, and blls_solve_reverse_a_prod re-entered with all other arguments unchanged. If the product cannot be formed, v need not be set, but blls_solve_reverse_a_prod should be re-entered with eval_status set to a nonzero value. This value of status can only occur if the user has set control.preconditioner = 2.

	*
		- eval_status

		- is a scalar variable of type int, that is used to indicate if the matrix products can be provided (see ``status`` above)

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the constant term :math:`b` in the residuals. The i-th component of b, i = 0, ... , m-1, contains :math:`b_i`.

	*
		- x_l

		- is a one-dimensional array of size n and type double, that holds the lower bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_l, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x_u

		- is a one-dimensional array of size n and type double, that holds the upper bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_u, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the values of the residuals :math:`c = A x - b`. The i-th component of c, i = 0, ... , m-1, contains :math:`c_i`.

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the values of the gradient :math:`g = A^T c`. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- z

		- is a one-dimensional array of size n and type double, that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type int, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- v

		- is a one-dimensional array of size n and type double, that is used for reverse communication (see status=2-4 above for details).

	*
		- p

		- is a one-dimensional array of size n and type double, that is used for reverse communication (see status=2-4 above for details).

	*
		- nz_v

		- is a one-dimensional array of size n and type int, that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_v_start

		- is a scalar of type int, that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_v_end

		- is a scalar of type int, that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_p

		- is a one-dimensional array of size n and type int, that is used for reverse communication (see status=4 above for details).

	*
		- nz_p_end

		- is a scalar of type int, that is used for reverse communication (see status=4 above for details).

	*
		- w

		- is an optional one-dimensional array of size m and type double, that holds the values :math:`w` of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

.. index:: pair: function; blls_information
.. _doxid-galahad__blls_8h_1a457b8ee7c630715bcb43427f254b555f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_information(void** data, struct :ref:`blls_inform_type<doxid-structblls__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`blls_inform_type <doxid-structblls__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; blls_terminate
.. _doxid-galahad__blls_8h_1ade863ffb6b142bfce669729f56911ac1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void blls_terminate(
		void** data,
		struct :ref:`blls_control_type<doxid-structblls__control__type>`* control,
		struct :ref:`blls_inform_type<doxid-structblls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`blls_inform_type <doxid-structblls__inform__type>`)
