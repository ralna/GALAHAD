.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_bqp_control_type.rst
	struct_bqp_inform_type.rst
	struct_bqp_time_type.rst

function calls
--------------

.. index:: pair: function; bqp_initialize
.. _doxid-galahad__bqp_8h_1a4466621895dd2314f1b3c21b4bc7f615:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_initialize(void** data, structure :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; bqp_read_specfile
.. _doxid-galahad__bqp_8h_1a0e3ffdd29be95753292694c7619a43e6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_read_specfile(struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, const Vararg{Cchar} specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNBQP.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/bqp.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; bqp_import
.. _doxid-galahad__bqp_8h_1a0cfa65e832fd80e3dfcf9e0c65a69e56:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_import(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void** data,
		int* status,
		Int32 n,
		const Vararg{Cchar} H_type[],
		Int32 ne,
		const int H_row[],
		const int H_col[],
		const int H_ptr[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows' or 'diagonal' has been violated.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to the Hessian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type Int32 that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type Int32 that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type Int32 that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL

.. index:: pair: function; bqp_import_without_h
.. _doxid-galahad__bqp_8h_1a9a99d880b3bfbcfb7b093756019c5f0e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_import_without_h(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void** data,
		int* status,
		Int32 n
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 has been violated.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables.

.. index:: pair: function; bqp_reset_control
.. _doxid-galahad__bqp_8h_1a315ce83042f67a466cfdd868c27a2850:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_reset_control(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; bqp_solve_given_h
.. _doxid-galahad__bqp_8h_1acb5ad644890efe38b7cf7048d6297308:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_solve_given_h(
		void** data,
		int* status,
		Int32 n,
		Int32 h_ne,
		const T H_val[],
		const T g[],
		const T f,
		const T x_l[],
		const T x_u[],
		T x[],
		T z[],
		Int32 x_stat[]
	)

Solve the bound-constrained quadratic program when the Hessian :math:`H` is available.



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

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful.



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows' or 'diagonal' has been violated.

		  * -4. The simple-bound constraints are inconsistent.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -20. The Hessian matrix :math:`H` appears to be indefinite. specified.

		  * -23. An entry from the strict upper triangle of :math:`H` has been

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

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
		- x_l

		- is a one-dimensional array of size n and type T that holds the lower bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_l, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the upper bounds :math:`x^l` on the variables :math:`x`. The j-th component of x_u, j = 0, ... , n-1, contains :math:`x^l_j`.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type Int32 that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; bqp_solve_reverse_h_prod
.. _doxid-galahad__bqp_8h_1a116b9b4ff28b9e2d18be0f0900ce2755:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_solve_reverse_h_prod(
		void** data,
		int* status,
		Int32 n,
		const T g[],
		const T f,
		const T x_l[],
		const T x_u[],
		T x[],
		T z[],
		Int32 x_stat[],
		T v[],
		const T prod[],
		Int32 nz_v[],
		int* nz_v_start,
		int* nz_v_end,
		const int nz_prod[],
		Int32 nz_prod_end
	)

Solve the bound-constrained quadratic program when the products of the Hessian :math:`H` with specified vectors may be computed by the calling program.



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

		  * -3. The restriction n > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows' or 'diagonal' has been violated.

		  * -4. The simple-bound constraints are inconsistent.

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -17. The step is too small to make further impact.

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -20. The Hessian matrix :math:`H` appears to be indefinite. specified.

		  * -23. An entry from the strict upper triangle of :math:`H` has been specified.

		  * 2. The product :math:`Hv` of the Hessian :math:`H` with a given output vector :math:`v` is required from the user. The vector :math:`v` will be stored in v and the product :math:`Hv` must be returned in prod, and bqp_solve_reverse_h_prod re-entered with all other arguments unchanged.

		  * 3. The product :math:`Hv` of the Hessian H with a given output vector :math:`v` is required from the user. Only components nz_v[nz_v_start-1:nz_v_end-1] of the vector :math:`v` stored in v are nonzero. The resulting product :math:`Hv` must be placed in prod, and bqp_solve_reverse_h_prod re-entered with all other arguments unchanged.

		  * 4. The product :math:`Hv` of the Hessian H with a given output vector :math:`v` is required from the user. Only components nz_v[nz_v_start-1:nz_v_end-1] of the vector :math:`v` stored in v are nonzero. The resulting **nonzeros** in the product :math:`Hv` must be placed in their appropriate comnponents of prod, while a list of indices of the nonzeros placed in nz_prod[0 : nz_prod_end-1]. bqp_solve_reverse_h_prod should then be re-entered with all other arguments unchanged. Typically v will be very sparse (i.e., nz_p_end-nz_p_start will be small).

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term :math:`g` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- f

		- is a scalar of type T that holds the constant term :math:`f` of the objective function.

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
		- z

		- is a one-dimensional array of size n and type T that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type Int32 that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=2-4 above for details)

	*
		- prod

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=2-4 above for details)

	*
		- nz_v

		- is a one-dimensional array of size n and type Int32 that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_v_start

		- is a scalar of type Int32 that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_v_end

		- is a scalar of type Int32 that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_prod

		- is a one-dimensional array of size n and type Int32 that is used for reverse communication (see status=4 above for details)

	*
		- nz_prod_end

		- is a scalar of type Int32 that is used for reverse communication (see status=4 above for details)

.. index:: pair: function; bqp_information
.. _doxid-galahad__bqp_8h_1a75b662635f281148e9c19e12e0788362:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_information(void** data, structure :ref:`bqp_inform_type<doxid-structbqp__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`bqp_inform_type <doxid-structbqp__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; bqp_terminate
.. _doxid-galahad__bqp_8h_1a34db499197d1fd6fb78b294473796fbc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void bqp_terminate(
		void** data,
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		struct :ref:`bqp_inform_type<doxid-structbqp__inform__type>`* inform
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

		- is a structure containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`bqp_inform_type <doxid-structbqp__inform__type>`)
