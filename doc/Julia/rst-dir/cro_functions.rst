.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_cro_control_type.rst
	struct_cro_inform_type.rst
	struct_cro_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`cro_control_type<doxid-structcro__control__type>`;
	struct :ref:`cro_inform_type<doxid-structcro__inform__type>`;
	struct :ref:`cro_time_type<doxid-structcro__time__type>`;

	// function calls

	void :ref:`cro_initialize<doxid-galahad__cro_8h_1aeb10643b5d27efef952b60d9ba0eb206>`(void** data, struct :ref:`cro_control_type<doxid-structcro__control__type>`* control, int* status);
	void :ref:`cro_read_specfile<doxid-galahad__cro_8h_1a55c7770ae26847b5c17055c290a54c2a>`(struct :ref:`cro_control_type<doxid-structcro__control__type>`* control, const char specfile[]);

	void :ref:`cro_crossover_solution<doxid-galahad__cro_8h_1a1ab8bdd6e394fe4d89c1c2acba8a5a7b>`(
		void** data,
		struct :ref:`cro_control_type<doxid-structcro__control__type>`* control,
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>`* inform,
		int n,
		int m,
		int m_equal,
		int h_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` H_val[],
		const int H_col[],
		const int H_ptr[],
		int a_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const int A_col[],
		const int A_ptr[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_u[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		int x_stat[],
		int c_stat[]
	);

	void :ref:`cro_terminate<doxid-galahad__cro_8h_1ae0692951f03b0999f73a8f68b7d62212>`(
		void** data,
		struct :ref:`cro_control_type<doxid-structcro__control__type>`* control,
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>`* inform
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

.. index:: pair: function; cro_initialize
.. _doxid-galahad__cro_8h_1aeb10643b5d27efef952b60d9ba0eb206:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void cro_initialize(void** data, struct :ref:`cro_control_type<doxid-structcro__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`cro_control_type <doxid-structcro__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The initialization was succesful.

.. index:: pair: function; cro_read_specfile
.. _doxid-galahad__cro_8h_1a55c7770ae26847b5c17055c290a54c2a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void cro_read_specfile(struct :ref:`cro_control_type<doxid-structcro__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNCRO.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/cro.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`cro_control_type <doxid-structcro__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; cro_crossover_solution
.. _doxid-galahad__cro_8h_1a1ab8bdd6e394fe4d89c1c2acba8a5a7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void cro_crossover_solution(
		void** data,
		struct :ref:`cro_control_type<doxid-structcro__control__type>`* control,
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>`* inform,
		int n,
		int m,
		int m_equal,
		int h_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` H_val[],
		const int H_col[],
		const int H_ptr[],
		int a_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const int A_col[],
		const int A_ptr[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` g[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c_u[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_l[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x_u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` c[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` y[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` z[],
		int x_stat[],
		int c_stat[]
	)

Crosover the solution from a primal-dual to a basic one.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`cro_control_type <doxid-structcro__control__type>`). The parameter .status is as follows:

	*
		- data

		- holds private internal data.

	*
		- inform

		- 
		  is a struct containing output information (see :ref:`cro_inform_type <doxid-structcro__inform__type>`). The component .status gives the exit status from the package. Possible values are:
		  
		  * 0. The crossover was succesful.
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 or m >= m_equal >= 0 has been violated.
		  
		  * -4 the bound constraints are inconsistent.
		  
		  * -5 the general constraints are likely inconsistent.
		  
		  * -9 an error has occured in SLS_analyse.
		  
		  * -10 an error has occured in SLS_factorize.
		  
		  * -11 an error has occured in SLS_solve.
		  
		  * -12 an error has occured in ULS_factorize.
		  
		  * -14 an error has occured in ULS_solve.
		  
		  * -16 the residuals are large; the factorization may be unsatisfactory.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables.

	*
		- m

		- is a scalar variable of type int, that holds the number of general linear constraints.

	*
		- m_equal

		- is a scalar variable of type int, that holds the number of general linear equality constraints. Such constraints must occur first in :math:`A`.

	*
		- h_ne

		- is a scalar variable of type int, that holds the number of entries in the **lower triangular** part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H`. The entries are stored by consecutive rows, the order within each row is unimportant.

	*
		- H_col

		- is a one-dimensional array of type int, that holds the column indices of the lower triangular part of :math:`H`, in the same order as those in H_val.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of :math:`H`. The n+1-st component holds the total number of entries (plus one if fortran indexing is used).

	*
		- a_ne

		- is a scalar variable of type int, that holds the number of entries in the constraint Jacobian matrix :math:`A`.

	*
		- A_val

		- is a one-dimensional array of type double, that holds the values of the entries of the constraint Jacobian matrix :math:`A`. The entries are stored by consecutive rows, the order within each row is unimportant. **Equality constraints must be ordered first.**

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of :math:`A` in the same order as those in A_val.

	*
		- A_ptr

		- is a one-dimensional array of size m+1 and type int, that holds the starting position of each row of :math:`A`. The m+1-st component holds the total number of entries (plus one if fortran indexing is used).

	*
		- g

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`g` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- c_l

		- is a one-dimensional array of size m and type double, that holds the lower bounds :math:`c^l` on the constraints :math:`A x`. The i-th component of c_l, i = 0, ... , m-1, contains :math:`c^l_i`.

	*
		- c_u

		- is a one-dimensional array of size m and type double, that holds the upper bounds :math:`c^l` on the constraints :math:`A x`. The i-th component of c_u, i = 0, ... , m-1, contains :math:`c^u_i`.

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

		- is a one-dimensional array of size m and type double, that holds the residual :math:`c(x) = A x`. The i-th component of c, j = 0, ... , n-1, contains :math:`c_j(x)`.

	*
		- y

		- is a one-dimensional array of size n and type double, that holds the values :math:`y` of the Lagrange multipliers for the general linear constraints. The j-th component of y, j = 0, ... , n-1, contains :math:`y_j`.

	*
		- z

		- is a one-dimensional array of size n and type double, that holds the values :math:`z` of the dual variables. The j-th component of z, j = 0, ... , n-1, contains :math:`z_j`.

	*
		- x_stat

		- is a one-dimensional array of size n and type int, that must be set on entry to give the status of the problem variables. If x_stat(j) is negative, the variable :math:`x_j` is active on its lower bound, if it is positive, it is active and lies on its upper bound, and if it is zero, it is inactiive and lies between its bounds. On exit, the :math:`j` -th component of x_stat is -1 if the variable is basic and active on its lower bound, -2 it is non-basic but active on its lower bound, 1 if it is basic and active on its upper bound, 2 it is non-basic but active on its upper bound, and 0 if it is inactive.

	*
		- c_stat

		- is a one-dimensional array of size m and type int, that must be set on entry to give the status of the general linear constraints. If c_stat(i) is negative, the constraint value :math:`a_i^Tx` is active on its lower bound, if it is positive, it is active and lies on its upper bound, and if it is zero, it is inactiive and lies between its bounds. On exit, the :math:`i` -th component of x_stat is -1 if the constraint is basic and active on its lower bound, -2 it is non-basic but active on its lower bound, 1 if it is basic and active on its upper bound, 2 it is non-basic but active on its upper bound, and 0 if it is inactive.

.. index:: pair: function; cro_terminate
.. _doxid-galahad__cro_8h_1ae0692951f03b0999f73a8f68b7d62212:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void cro_terminate(
		void** data,
		struct :ref:`cro_control_type<doxid-structcro__control__type>`* control,
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`cro_control_type <doxid-structcro__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`cro_inform_type <doxid-structcro__inform__type>`)

