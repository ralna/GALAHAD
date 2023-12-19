.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_clls_control_type.rst
	struct_clls_inform_type.rst
	struct_clls_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`clls_control_type<doxid-structclls__control__type>`;
	struct :ref:`clls_inform_type<doxid-structclls__inform__type>`;
	struct :ref:`clls_time_type<doxid-structclls__time__type>`;

	// function calls

	void :ref:`clls_initialize<doxid-galahad__clls_8h_1a782387ad9cccc5f2e2da9df9016fb923>`(void** data, struct :ref:`clls_control_type<doxid-structclls__control__type>`* control, int* status);
	void :ref:`clls_read_specfile<doxid-galahad__clls_8h_1ade439e5e06c2852fcb089bb39a667a74>`(struct :ref:`clls_control_type<doxid-structclls__control__type>`* control, const char specfile[]);

	void :ref:`clls_import<doxid-galahad__clls_8h_1a6a2be17b6f871df80bbac93940b83af3>`(
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		void** data,
		int* status,
		int n,
		int o,
		int m,
		const char Ao_type[],
		int Ao_ne,
		const int Ao_row[],
		const int Ao_col[],
		const int Ao_ptr[],
		const char A_type[],
		int A_ne,
		const int A_row[],
		const int A_col[],
		const int A_ptr[]
	);

	void :ref:`clls_reset_control<doxid-galahad__clls_8h_1a9f7ccb0cffa909a2be7556edda430190>`(
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`clls_solve_clls<doxid-galahad__clls_8h_1ac2d720ee7b719bf63c3fa208d37f1bc1>`(
		void** data,
		int* status,
		int n,
		int o,
		int m,
		int ao_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		int a_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_u[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` r[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		int x_stat[],
		int c_stat[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` regularization_weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` w[]
	);

	void :ref:`clls_information<doxid-galahad__clls_8h_1adfb7589696e4e07fdb65f02bc42c5daf>`(void** data, struct :ref:`clls_inform_type<doxid-structclls__inform__type>`* inform, int* status);

	void :ref:`clls_terminate<doxid-galahad__clls_8h_1a84e12e9e546f51762d305333dce68e2b>`(
		void** data,
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		struct :ref:`clls_inform_type<doxid-structclls__inform__type>`* inform
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

.. index:: pair: function; clls_initialize
.. _doxid-galahad__clls_8h_1a782387ad9cccc5f2e2da9df9016fb923:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_initialize(void** data, struct :ref:`clls_control_type<doxid-structclls__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; clls_read_specfile
.. _doxid-galahad__clls_8h_1ade439e5e06c2852fcb089bb39a667a74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_read_specfile(struct :ref:`clls_control_type<doxid-structclls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/clls/CLLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/clls.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; clls_import
.. _doxid-galahad__clls_8h_1a6a2be17b6f871df80bbac93940b83af3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_import(
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		void** data,
		int* status,
		int n,
		int o,
		int m,
		const char Ao_type[],
		int Ao_ne,
		const int Ao_row[],
		const int Ao_col[],
		const int Ao_ptr[],
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

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
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
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.
		  
	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- o

		- is a scalar variable of type int, that holds the number of residulas.

	*
		- m

		- is a scalar variable of type int, that holds the number of general linear constraints.

	*
		- Ao_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the objective design matrix, $A_o$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- Ao_ne

		- is a scalar variable of type int, that holds the number of entries in $A_o$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- Ao_row

		- is a one-dimensional array of size A_ne and type int, that holds the row indices of $A_o$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- Ao_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- Ao_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of $A_o$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type int, that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; clls_reset_control
.. _doxid-galahad__clls_8h_1a9f7ccb0cffa909a2be7556edda430190:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_reset_control(
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful.

.. index:: pair: function; clls_solve_qp
.. _doxid-galahad__clls_8h_1ac2d720ee7b719bf63c3fa208d37f1bc1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_solve_qp(
		void** data,
		int* status,
		int n,
		int o,
		int m,
		int ao_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` Ao_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		int a_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_u[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` r[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		int x_stat[],
		int c_stat[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` regularization_weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` w[]
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
		  is a scalar variable of type int, that gives the entry and exit status from the package.
		  
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

		- is a scalar variable of type int, that holds the number of variables.

	*
		- o

		- is a scalar variable of type int, that holds the number of residulas.

	*
		- m

		- is a scalar variable of type int, that holds the number of general linear constraints.


	*
		- ao_ne

		- is a scalar variable of type int, that holds the number of entries in the objective design matrix $A$.

	*
		- Ao_val

		- is a one-dimensional array of size ao_ne and type double, that holds the values of the entries of the objective design Jacobian matrix $A$ in any of the available storage schemes.


	*
		- h_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the linear term $b$ of observations. The j-th component of b, i = 0, ... , m-1, contains $b_i$.

	*
		- a_ne

		- is a scalar variable of type int, that holds the number of entries in the constraint Jacobian matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type double, that holds the values of the entries of the constraint Jacobian matrix $A$ in any of the available storage schemes.

	*
		- c_l

		- is a one-dimensional array of size m and type double, that holds the lower bounds $c^l$ on the constraints $A x$. The i-th component of c_l, i = 0, ... , m-1, contains $c^l_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type double, that holds the upper bounds $c^l$ on the constraints $A x$. The i-th component of c_u, i = 0, ... , m-1, contains $c^u_i$.

	*
		- x_l

		- is a one-dimensional array of size n and type double, that holds the lower bounds $x^l$ on the variables $x$. The j-th component of x_l, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type double, that holds the upper bounds $x^l$ on the variables $x$. The j-th component of x_u, j = 0, ... , n-1, contains $x^l_j$.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size o and type double, that holds the residuals $r(x)=A_0x-b$. The i-th component of r, i = 0, ... , o-1, contains $r_i$.

	*
		- c

		- is a one-dimensional array of size m and type double, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type double, that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , n-1, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type double, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type int, that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type int, that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- regularization_weight

		- is a scalar of type double, that holds the  non-negative regularization weight $\sigma \geq 0$.

	*
		- w

		- is a one-dimensional array of size o and type double, that holds the values $w$ of strictly-positive observation weights. The i-th component of w, j = 0, ... , o-1, contains $w_i$. If the weights are all one, w can be set to NULL.

.. index:: pair: function; clls_information
.. _doxid-galahad__clls_8h_1adfb7589696e4e07fdb65f02bc42c5daf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_information(void** data, struct :ref:`clls_inform_type<doxid-structclls__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`clls_inform_type <doxid-structclls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; clls_terminate
.. _doxid-galahad__clls_8h_1a84e12e9e546f51762d305333dce68e2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void clls_terminate(
		void** data,
		struct :ref:`clls_control_type<doxid-structclls__control__type>`* control,
		struct :ref:`clls_inform_type<doxid-structclls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`clls_inform_type <doxid-structclls__inform__type>`)

