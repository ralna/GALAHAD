.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_bqp_control_type.rst
	struct_bqp_inform_type.rst
	struct_bqp_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`bqp_control_type<doxid-structbqp__control__type>`;
	struct :ref:`bqp_inform_type<doxid-structbqp__inform__type>`;
	struct :ref:`bqp_time_type<doxid-structbqp__time__type>`;

	// function calls

	void :ref:`bqp_initialize<doxid-galahad__bqp_8h_1a4466621895dd2314f1b3c21b4bc7f615>`(void **data, struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`bqp_read_specfile<doxid-galahad__bqp_8h_1a0e3ffdd29be95753292694c7619a43e6>`(struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, const char specfile[]);

	void :ref:`bqp_import<doxid-galahad__bqp_8h_1a0cfa65e832fd80e3dfcf9e0c65a69e56>`(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	);

	void :ref:`bqp_import_without_h<doxid-galahad__bqp_8h_1a9a99d880b3bfbcfb7b093756019c5f0e>`(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n
	);

	void :ref:`bqp_reset_control<doxid-galahad__bqp_8h_1a315ce83042f67a466cfdd868c27a2850>`(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`bqp_solve_given_h<doxid-galahad__bqp_8h_1acb5ad644890efe38b7cf7048d6297308>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[]
	);

	void :ref:`bqp_solve_reverse_h_prod<doxid-galahad__bqp_8h_1a116b9b4ff28b9e2d18be0f0900ce2755>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` prod[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nz_v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *nz_v_start,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *nz_v_end,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` nz_prod[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nz_prod_end
	);

	void :ref:`bqp_information<doxid-galahad__bqp_8h_1a75b662635f281148e9c19e12e0788362>`(void **data, struct :ref:`bqp_inform_type<doxid-structbqp__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`bqp_terminate<doxid-galahad__bqp_8h_1a34db499197d1fd6fb78b294473796fbc>`(
		void **data,
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		struct :ref:`bqp_inform_type<doxid-structbqp__inform__type>`* inform
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

.. index:: pair: function; bqp_initialize
.. _doxid-galahad__bqp_8h_1a4466621895dd2314f1b3c21b4bc7f615:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_initialize(void **data, struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; bqp_read_specfile
.. _doxid-galahad__bqp_8h_1a0e3ffdd29be95753292694c7619a43e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_read_specfile(struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values 
associated with given keywords to the corresponding control 
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/bqp/BQP.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/bqp.pdf for a list of how these keywords relate 
to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; bqp_import
.. _doxid-galahad__bqp_8h_1a0cfa65e832fd80e3dfcf9e0c65a69e56:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_import(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to the Hessian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL

.. index:: pair: function; bqp_import_without_h
.. _doxid-galahad__bqp_8h_1a9a99d880b3bfbcfb7b093756019c5f0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_import_without_h(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

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
		  
		  * -3. The restriction n > 0 has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

.. index:: pair: function; bqp_reset_control
.. _doxid-galahad__bqp_8h_1a315ce83042f67a466cfdd868c27a2850:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_reset_control(
		struct :ref:`bqp_control_type<doxid-structbqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

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

.. index:: pair: function; bqp_solve_given_h
.. _doxid-galahad__bqp_8h_1acb5ad644890efe38b7cf7048d6297308:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_solve_given_h(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[]
	)

Solve the bound-constrained quadratic program when the Hessian $H$ is available.



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
		  
		  On initial entry, status must be set to 1.
		  
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
                    The restriction n > 0 or requirement that a type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows' or 'diagonal' has been violated.
		  
		  * **-4**
                    The simple-bound constraints are inconsistent.
		  
		  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status
		  
		  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.
		  
		  * **-11**
                    The solution of a set of linear equations using
                    factors from the factorization package failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.
		  
		  * **-17**
                    The step is too small to make further impact.
		  
		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-20**
                    The Hessian matrix $H$ appears to be
                    indefinite. specified.
		  
		  * **-23**
                    An entry from the strict upper triangle of $H$
                    has been

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- h_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $g$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- f

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the constant term $f$ of the objective function.

	*
		- x_l

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the lower bounds $x^l$ on the variables $x$. The j-th component of x_l, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the upper bounds $x^l$ on the variables $x$. The j-th component of x_u, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; bqp_solve_reverse_h_prod
.. _doxid-galahad__bqp_8h_1a116b9b4ff28b9e2d18be0f0900ce2755:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_solve_reverse_h_prod(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` prod[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nz_v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *nz_v_start,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *nz_v_end,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` nz_prod[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` nz_prod_end
	)

Solve the bound-constrained quadratic program when the products of the Hessian $H$ with specified vectors may be computed by the calling program.



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
		  
		  * 0. The run was successful.
		  
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
                    The restriction n > 0 or requirement that a type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows' or 'diagonal' has been violated.
		  
		  * **-4**
                    The simple-bound constraints are inconsistent.
		  
		  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status
		  
		  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.
		  
		  * **-11**
                    The solution of a set of linear equations using
                    factors from the factorization package failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.
		  
		  * **-17**
                    The step is too small to make further impact.
		  
		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-20**
                    The Hessian matrix $H$ appears to be
                    indefinite. specified.
		  
		  * **-23**
                    An entry from the strict upper triangle of $H$
                    has been specified.

		  * **2**
                    The product $Hv$ of the Hessian $H$ with
                    a given output vector $v$ is required from the
                    user. The vector $v$ will be stored in v and
                    the product $Hv$ must be returned in prod, and
                    bqp_solve_reverse_h_prod re-entered with all other
                    arguments unchanged.
		  
		  * **3**
                    The product $Hv$ of the Hessian H with a given
                    output vector $v$ is required from the
                    user. Only components nz_v[nz_v_start-1:nz_v_end-1]
                    of the vector $v$ stored in v are nonzero. The
                    resulting product $Hv$ must be placed in prod,
                    and bqp_solve_reverse_h_prod re-entered with all
                    other arguments unchanged.
		  
		  * **4**
                    The product $Hv$ of the Hessian H with a given
                    output vector $v$ is required from the
                    user. Only components nz_v[nz_v_start-1:nz_v_end-1]
                    of the vector $v$ stored in v are nonzero. The
                    resulting **nonzeros** in the product $Hv$
                    must be placed in their appropriate comnponents of
                    prod, while a list of indices of the nonzeros placed
                    in nz_prod[0 :
                    nz_prod_end-1]. bqp_solve_reverse_h_prod should then
                    be re-entered with all other arguments
                    unchanged. Typically v will be very sparse (i.e.,
                    nz_p_end-nz_p_start will be small).

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $g$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- f

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the constant term $f$ of the objective function.

	*
		- x_l

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the lower bounds $x^l$ on the variables $x$. The j-th component of x_l, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the upper bounds $x^l$ on the variables $x$. The j-th component of x_u, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- v

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication (see status=2-4 above for details)

	*
		- prod

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication (see status=2-4 above for details)

	*
		- nz_v

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_v_start

		- is a scalar of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_v_end

		- is a scalar of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication (see status=3-4 above for details)

	*
		- nz_prod

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication (see status=4 above for details)

	*
		- nz_prod_end

		- is a scalar of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication (see status=4 above for details)

.. index:: pair: function; bqp_information
.. _doxid-galahad__bqp_8h_1a75b662635f281148e9c19e12e0788362:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_information(void **data, struct :ref:`bqp_inform_type<doxid-structbqp__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`bqp_inform_type <doxid-structbqp__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; bqp_terminate
.. _doxid-galahad__bqp_8h_1a34db499197d1fd6fb78b294473796fbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bqp_terminate(
		void **data,
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

		- is a struct containing control information (see :ref:`bqp_control_type <doxid-structbqp__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`bqp_inform_type <doxid-structbqp__inform__type>`)

