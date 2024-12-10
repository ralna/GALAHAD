.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_dqp_control_type.rst
	struct_dqp_inform_type.rst
	struct_dqp_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`dqp_control_type<doxid-structdqp__control__type>`;
	struct :ref:`dqp_inform_type<doxid-structdqp__inform__type>`;
	struct :ref:`dqp_time_type<doxid-structdqp__time__type>`;

	// function calls

	void :ref:`dqp_initialize<doxid-galahad__dqp_8h_1a19aea950ca15a63e11702af3b4e777a2>`(void **data, struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`dqp_read_specfile<doxid-galahad__dqp_8h_1a1db755c043c56f0afdc8e61c8ebfc517>`(struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control, const char specfile[]);

	void :ref:`dqp_import<doxid-galahad__dqp_8h_1a126153a2c845e1840b01cbd28a5b187d>`(
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[]
	);

	void :ref:`dqp_reset_control<doxid-galahad__dqp_8h_1abff958ca01c88bb049bd06a238dcbefe>`(
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`dqp_solve_qp<doxid-galahad__dqp_8h_1a2b72dcb3fe12c15b79be741304583bd4>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_stat[]
	);

	void :ref:`dqp_solve_sldqp<doxid-galahad__dqp_8h_1a5175cda6f83c34b45115527af26f9da2>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x0[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_stat[]
	);

	void :ref:`dqp_information<doxid-galahad__dqp_8h_1ae0ac5195553f6dcccc43f53f0e08b0a4>`(void **data, struct :ref:`dqp_inform_type<doxid-structdqp__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`dqp_terminate<doxid-galahad__dqp_8h_1ac2f0f150bf38c9cc0ea33f91df913d1a>`(
		void **data,
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		struct :ref:`dqp_inform_type<doxid-structdqp__inform__type>`* inform
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

.. index:: pair: function; dqp_initialize
.. _doxid-galahad__dqp_8h_1a19aea950ca15a63e11702af3b4e777a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_initialize(void **data, struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`dqp_control_type <doxid-structdqp__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; dqp_read_specfile
.. _doxid-galahad__dqp_8h_1a1db755c043c56f0afdc8e61c8ebfc517:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_read_specfile(struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/dqp/DQP.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/dqp.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`dqp_control_type <doxid-structdqp__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; dqp_import
.. _doxid-galahad__dqp_8h_1a126153a2c845e1840b01cbd28a5b187d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_import(
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`dqp_control_type <doxid-structdqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful
		  
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
                    The restrictions n > 0 or m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity' or 'identity' has been violated.
		  
		  * **-23**
                    An entry from the strict upper triangle of $H$
                    has been specified.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of general linear constraints.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', or 'identity'; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; dqp_reset_control
.. _doxid-galahad__dqp_8h_1abff958ca01c88bb049bd06a238dcbefe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_reset_control(
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`dqp_control_type <doxid-structdqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful.

.. index:: pair: function; dqp_solve_qp
.. _doxid-galahad__dqp_8h_1a2b72dcb3fe12c15b79be741304583bd4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_solve_qp(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_stat[]
	)

Solve the quadratic program when the Hessian $H$ is available.



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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.
		  
		  * **-5**
                    The simple-bound constraints are inconsistent.
		  
		  * **-7**
                    The constraints appear to have no feasible point.
		  
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
		  
		  * **-23**
                    An entry from the strict upper triangle of $H$
                    has been specified.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of general linear constraints.

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
		- a_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the constraint Jacobian matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the constraint Jacobian matrix $A$ in any of the available storage schemes.

	*
		- c_l

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the lower bounds $c^l$ on the constraints $A x$. The i-th component of c_l, i = 0, ... , m-1, contains $c^l_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the upper bounds $c^l$ on the constraints $A x$. The i-th component of c_u, i = 0, ... , m-1, contains $c^u_i$.

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
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , n-1, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; dqp_solve_sldqp
.. _doxid-galahad__dqp_8h_1a5175cda6f83c34b45115527af26f9da2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_solve_sldqp(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x0[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_stat[]
	)

Solve the shifted least-distance quadratic program



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
		  
		  * **0**
                    The run was successful
		  
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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.
		  
		  * **-5**
                    The simple-bound constraints are inconsistent.
		  
		  * **-7**
                    The constraints appear to have no feasible point.
		  
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
		  
		  * **-23**
                    An entry from the strict upper triangle of $H$
                    has been specified.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of general linear constraints.

	*
		- w

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the weights $w$.

	*
		- x0

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the shifts $x^0$.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $g$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- f

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the constant term $f$ of the objective function.

	*
		- a_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the constraint Jacobian matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the constraint Jacobian matrix $A$ in any of the available storage schemes.

	*
		- c_l

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the lower bounds $c^l$ on the constraints $A x$. The i-th component of c_l, i = 0, ... , m-1, contains $c^l_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the upper bounds $c^l$ on the constraints $A x$. The i-th component of c_u, i = 0, ... , m-1, contains $c^u_i$.

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
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , n-1, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; dqp_information
.. _doxid-galahad__dqp_8h_1ae0ac5195553f6dcccc43f53f0e08b0a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_information(void **data, struct :ref:`dqp_inform_type<doxid-structdqp__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`dqp_inform_type <doxid-structdqp__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; dqp_terminate
.. _doxid-galahad__dqp_8h_1ac2f0f150bf38c9cc0ea33f91df913d1a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dqp_terminate(
		void **data,
		struct :ref:`dqp_control_type<doxid-structdqp__control__type>`* control,
		struct :ref:`dqp_inform_type<doxid-structdqp__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`dqp_control_type <doxid-structdqp__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`dqp_inform_type <doxid-structdqp__inform__type>`)

