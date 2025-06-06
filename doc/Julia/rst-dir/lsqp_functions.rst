callable functions
------------------

.. index:: pair: function; lsqp_initialize
.. _doxid-galahad__lsqp_8h_1aac395e385cae77c266ec108f21e9e8f9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsqp_control_type <doxid-structlsqp__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; lsqp_read_specfile
.. _doxid-galahad__lsqp_8h_1a830242147779223fa2dbed69c2c0c200:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/lsqp/LSQP.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/lsqp.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lsqp_control_type <doxid-structlsqp__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; lsqp_import
.. _doxid-galahad__lsqp_8h_1a5b6f76c31025aa6794e81715f0362e70:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_import(T, INT, control, data, status, n, m, 
                             A_type, A_ne, A_row, A_col, A_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`lsqp_control_type <doxid-structlsqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

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

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-a_storage__unsym>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type INT that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type INT that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type INT that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; lsqp_reset_control
.. _doxid-galahad__lsqp_8h_1a3dc0d9ed7fad6f3ea575e1a53c06c35e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`lsqp_control_type <doxid-structlsqp__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; lsqp_solve_qp
.. _doxid-galahad__lsqp_8h_1a44019540a174679eda6d46b1ddae89f8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_solve_qp(T, INT, data, status, n, m, w, x0, g, f, a_ne, A_val, 
                               c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)

Solve the separable convex quadratic program.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the
		  entry and exit status from the package.

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

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- w

		- is a one-dimensional array of size n and type T that holds the values of the weights $w$.

	*
		- x0

		- is a one-dimensional array of size n and type T that holds the values of the shifts $x^0$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- f

		- is a scalar of type T that holds the constant term $f$ of the objective function.

	*
		- a_ne

		- is a scalar variable of type INT that holds the number of entries in the constraint Jacobian matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type T that holds the values of the entries of the constraint Jacobian matrix $A$ in any of the available storage schemes.

	*
		- c_l

		- is a one-dimensional array of size m and type T that holds the lower bounds $c^l$ on the constraints $A x$. The i-th component of ``c_l``, i = 1, ... , m, contains $c^l_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type T that holds the upper bounds $c^l$ on the constraints $A x$. The i-th component of ``c_u``, i = 1, ... , m, contains $c^u_i$.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the lower bounds $x^l$ on the variables $x$. The j-th component of ``x_l``, j = 1, ... , n, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the upper bounds $x^l$ on the variables $x$. The j-th component of ``x_u``, j = 1, ... , n, contains $x^l_j$.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, i = 1, ... , m, contains $c_i(x)$.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of ``y``, i = 1, ... , m, contains $y_i$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- c_stat

		- is a one-dimensional array of size m and type INT that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^T x$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

.. index:: pair: function; lsqp_information
.. _doxid-galahad__lsqp_8h_1aa86d9f9bfe75370d90a0be244a9a23ce:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_information(T, INT, data, inform, status)

Provides output information.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lsqp_inform_type <doxid-structlsqp__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lsqp_terminate
.. _doxid-galahad__lsqp_8h_1a7a9c9d7589c1acb11f0e2a579b1d8053:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsqp_terminate(T, INT, data, control, inform)

Deallocate all internal private storage.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsqp_control_type <doxid-structlsqp__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lsqp_inform_type <doxid-structlsqp__inform__type>`)
