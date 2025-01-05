callable functions
------------------

.. index:: pair: function; qpa_initialize
.. _doxid-galahad__qpa_8h_1afc82144e136ab34fe8a7aea4acd870fc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`qpa_control_type <doxid-structqpa__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; qpa_read_specfile
.. _doxid-galahad__qpa_8h_1a41a3965eddf14d030a6fa30795149369:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/qpa/QPA.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/qpa.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`qpa_control_type <doxid-structqpa__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; qpa_import
.. _doxid-galahad__qpa_8h_1a2ecf96b51001b1613ac0637c3bc44824:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_import(T, INT, control, data, status, n, m, 
                            H_type, H_ne, H_row, H_col, H_ptr,
                            A_type, A_ne, A_row, A_col, A_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`qpa_control_type <doxid-structqpa__control__type>`)

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

		  * **-23**
                    An entry from the strict upper triangle of $H$ has
                    been specified.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-ah_storage__sym>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if $H=0$; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type INT that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type INT that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be C_NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-ah_storage__unsym>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

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

.. index:: pair: function; qpa_reset_control
.. _doxid-galahad__qpa_8h_1a9d60441b2beaddb4c653156592ffc1ea:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`qpa_control_type <doxid-structqpa__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; qpa_solve_qp
.. _doxid-galahad__qpa_8h_1af9c60939ef803461d90631dd48cb55d7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_solve_qp(T, INT, data, status, n, m, h_ne, H_val, g, f, 
                              a_ne, A_val, c_l, c_u, x_l, x_u, 
                              x, c, y, z, x_stat, c_stat)

Solve the quadratic program (2)-(4).

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

		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-23**
                    An entry from the strict upper triangle of $H$ has
                    been specified.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

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

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the current status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, x_stat should be set as above to provide a guide to the initial working set.

	*
		- c_stat

		- is a one-dimensional array of size m and type INT that gives the current status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, c_stat should be set as above to provide a guide to the initial working set.

.. index:: pair: function; qpa_solve_l1qp
.. _doxid-galahad__qpa_8h_1a1a95adb548b743128e0df4ab4e801f19:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_solve_l1qp(T, INT, data, status, n, m, h_ne, H_val, g, f, 
                                 rho_g, rho_b, a_ne, A_val, c_l, c_u, 
                                 x_l, x_u, x, c, y, z, x_stat, c_stat)

Solve the l_1 quadratic program (1).

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

		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-23**
                    An entry from the strict upper triangle of $H$ has
                    been specified.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- f

		- is a scalar of type T that holds the constant term $f$ of the objective function.

	*
		- rho_g

		- is a scalar of type T that holds the parameter $\rho_g$ associated with the linear constraints.

	*
		- rho_b

		- is a scalar of type T that holds the parameter $\rho_b$ associated with the simple bound constraints.

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

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the current status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, x_stat should be set as above to provide a guide to the initial working set.

	*
		- c_stat

		- is a one-dimensional array of size m and type INT that gives the current status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, c_stat should be set as above to provide a guide to the initial working set.

.. index:: pair: function; qpa_solve_bcl1qp
.. _doxid-galahad__qpa_8h_1a5b5ef5f0d8134d8f02b1da62a04a3ace:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_solve_bcl1qp(T, INT, data, status, n, m, h_ne, H_val, g, f, 
                                  rho_g, a_ne, A_val, c_l, c_u, x_l, x_u, 
                                  x, c, y, z, x_stat, c_stat)

Solve the bound-constrained l_1 quadratic program (4)-(5)

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

		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-23**
                    An entry from the strict upper triangle of $H$ has
                    been specified.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- f

		- is a scalar of type T that holds the constant term $f$ of the objective function.

	*
		- rho_g

		- is a scalar of type T that holds the parameter $\rho_g$ associated with the linear constraints.

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

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the current status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, x_stat should be set as above to provide a guide to the initial working set.

	*
		- c_stat

		- is a one-dimensional array of size m and type INT that gives the current status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds. On entry, if control.cold_start = 0, c_stat should be set as above to provide a guide to the initial working set.

.. index:: pair: function; qpa_information
.. _doxid-galahad__qpa_8h_1a631cf6ec1a95b27c712ace4fa7dc06f0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`qpa_inform_type <doxid-structqpa__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; qpa_terminate
.. _doxid-galahad__qpa_8h_1a18b73b54796470edc039e3ac85bd30d5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function qpa_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`qpa_control_type <doxid-structqpa__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`qpa_inform_type <doxid-structqpa__inform__type>`)
