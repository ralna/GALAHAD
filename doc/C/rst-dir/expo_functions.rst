.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_expo_control_type.rst
	struct_expo_inform_type.rst
	struct_expo_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`expo_control_type<doxid-structexpo__control__type>`;
	struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`;
	struct :ref:`expo_time_type<doxid-structexpo__time__type>`;

	// function calls

	void :ref:`expo_initialize<doxid-galahad__expo_8h_1aa344bb15b74ab3b3ee6afb2de072b19f>`(
		void **data,
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform
	);

	void :ref:`expo_read_specfile<doxid-galahad__expo_8h_1adf9db7eff2fce137ae2abd2e013c47b3>`(struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control, const char specfile[]);

	void :ref:`expo_import<doxid-galahad__expo_8h_1a3f0eb83fd31ee4108156f2e84176389d>`(
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	);

	void :ref:`expo_reset_control<doxid-galahad__expo_8h_1a07f0857c9923ad0f92d51ed00833afda>`(
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`expo_solve_hessian_direct<doxid-galahad__expo_8h_1ae923c2e6afabb3563fe0998d45b715c4>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` gl[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], 
                 const void*) eval_fc,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], , :ref:`rpc_<doxid-galahad__rpc_8h_>`[], 
                 const void*) eval_gj,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], 
                 const void*) eval_hl
	);

	void :ref:`expo_information<doxid-galahad__expo_8h_1a765da96b0a1f3d07dab53cc3400c22d8>`(void **data, struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`expo_terminate<doxid-galahad__expo_8h_1a7babe9112dfad1eb7b57b70135704ab0>`(
		void **data,
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform
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

.. index:: pair: function; expo_initialize
.. _doxid-galahad__expo_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_initialize(
		void **data,
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)

.. index:: pair: function; expo_read_specfile
.. _doxid-galahad__expo_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_read_specfile(struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/expo/EXPO.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/expo.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; expo_import
.. _doxid-galahad__expo_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_import(
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

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
                    The restrictions n > 0, m $\geq$ 0 or requirement that
                    J/H_type contains its relevant string 'dense',
                    'dense_by_columns', 'coordinate', 'sparse_by_rows',
                    'sparse_by_columns', 'diagonal' or 'absent' has been
                    violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of constraints.

	*
		- J_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the Jacobian, $J$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- J_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- J_row

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- J_col

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $J$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- J_ptr

		- is a one-dimensional array of size m+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $J$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, $H_L$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to $H$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $H_L$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $H_L$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $H_L$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; expo_reset_control
.. _doxid-galahad__expo_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_reset_control(
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; expo_solve_hessian_direct
.. _doxid-galahad__expo_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_solve_hessian_direct(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c_u[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_l[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` x_u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` gl[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], 
                const void*) eval_c,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], 
                const void*) eval_j,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[],  
                const void*) eval_h,
	)

Find a local minimizer of a given constrained optimization problem.

This call is for the case where $H(x,y) = \nabla_{xx}f(x)
- \sum_i y_i \nabla_{xx}c_i(x)$ is provided specifically, and all function/derivative information is available by function calls.


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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  On initial entry, status must be set to 1.
		  
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
                    The restriction n > 0, m $\geq$ 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows', or 'diagonal' has been violated.
		  
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
                    happen if control.max_it or control.max_eval is too small, 
                    but may also be symptomatic of a badly scaled problem.
		  
		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of constraints.

	*
		- J_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $J$.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $H_L$.

	*
		- c_l

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $c_l$ of the lower bounds on the constraint functions $c(x)$. The i-th component of c_l, $i = 0, \ldots, m-1$, contains $c_{li}$.

	*
		- c_u

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $c_u$ of the upper bounds on the constraint functions $c(x)$. The i-th component of c_u, $i = 0, \ldots, m-1$, contains $c_{ui}$.

	*
		- x_l

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x_l$ of the lower bounds on the optimization variables $x$. The j-th component of x_l, $j = 0, \ldots, n-1$, contains $x_{lj}$.

	*
		- x_u

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x_u$ of the upper bounds on the optimization variables $x$. The j-th component of x_u, $j = 0, \ldots, n-1$, contains $x_{uj}$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$. This should be set on input to an estimate of the minimizer.

	*
		- y

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $y$ of the Lagrange multipliers. The j-th component of y, i = 0, ... , m-1, contains $y_i$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the constraints $c(x)$. The i-th component of c, i = 0, ... , n-1, contains $c_i(x)$.

	*
		- gl

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g_L(x,y)$ of the Lagrangian function. The j-th component of gl, j = 0, ... , n-1, contains $g_{Lj}$.


	*
		- eval_fc

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_fc( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` f, :ref:`rpc_<doxid-galahad__rpc_8h_>` c[], 
                          const void *userdata )
		  
		  The value of the objective function $f(x)$ and the components of the constraint function $c(x)$ evaluated at x= $x$ must be assigned to f and c, respectively, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_fc`` via the structure ``userdata``.

	*
		- eval_gj

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_gj( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` jne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` g[], 
                          :ref:`rpc_<doxid-galahad__rpc_8h_>` j[], const void *userdata )
		  
		  The components of the gradient $g = g(x)$ of the objective and Jacobian $J = \nabla_x c(x$) of the constraints must be assigned to g and to j, in the same order as presented to expo_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_gj`` via the structure ``userdata``.

	*
		- eval_hl

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_hl( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` hne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], 
                         const :ref:`rpc_<doxid-galahad__rpc_8h_>` y[], :ref:`rpc_<doxid-galahad__rpc_8h_>` h[], const void *userdata )
		  
		  The nonzeros of the matrix $H_L(x,y) = \nabla_{xx}f(x) -\sum_i y_i \nabla_{xx}c_i(x)$ of the Hessian of the Lagrangian function evaluated at x= $x$ and y= $y$ must be assigned to h in the same order as presented to expo_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_hl`` via the structure ``userdata``.
	
.. index:: pair: function; expo_information
.. _doxid-galahad__expo_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_information(void **data, struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; expo_terminate
.. _doxid-galahad__expo_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void expo_terminate(
		void **data,
		struct :ref:`expo_control_type<doxid-structexpo__control__type>`* control,
		struct :ref:`expo_inform_type<doxid-structexpo__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)

