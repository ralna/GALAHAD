.. _global:

callable functions
------------------

.. index:: pair: function; expo_initialize
.. _doxid-galahad__expo_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_initialize(T, INT, data, control, inform)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)

.. index:: pair: function; expo_read_specfile
.. _doxid-galahad__expo_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/expo/EXPO.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/expo.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; expo_import
.. _doxid-galahad__expo_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_import(T, INT, control, data, status, n, m,
                            J_type, J_ne, J_row, J_col, J_ptr,
                            H_type, H_ne, H_row, H_col, H_ptr )

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

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
                    The restrictions n > 0, m > 0 or requirement that
                    J/H_type contains its relevant string 'dense',
                    'dense_by_columns', 'coordinate', 'sparse_by_rows',
                    'sparse_by_columns', 'diagonal' or 'absent' has been
                    violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of general constraints.

	*
		- J_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-expo_storage__unsym>` used for the Jacobian, $J$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- J_ne

		- is a scalar variable of type INT that holds the number of entries in $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- J_row

		- is a one-dimensional array of size J_ne and type INT that holds the row indices of $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- J_col

		- is a one-dimensional array of size J_ne and type INT that holds the column indices of $J$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- J_ptr

		- is a one-dimensional array of size m+1 and type INT that holds the starting position of each row of $J$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <details-expo_storage__sym>` used for the Hessian, $H_L$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to $H$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H_L$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type INT that holds the row indices of the lower triangular part of $H_L$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type INT that holds the column indices of the lower triangular part of $H_L$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.


.. index:: pair: function; expo_reset_control
.. _doxid-galahad__expo_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; expo_solve_hessian_direct
.. _doxid-galahad__expo_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_solve_hessian_direct(T, INT, data, userdata, status, 
                                           n, m, j_ne, h_ne, 
                                           c_l, c_u, x_l, x_u, 
                                           x, y, z, c, gl,
                                           eval_fc, eval_gj, eval_hl)

Find a local minimizer of the constrained optimization problem using the
exponential penalty method.

This call is for the case where the Hessian of the Lagrangian function is
available specifically, and all function/derivative information is
available by (direct) function calls.

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

		- is a scalar variable of type INT that gives the
		  entry and exit status from the package.

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
                    The restriction n > 0 or requirement that type
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
                    Too many iterations have been performed. This may happen
                    if control.max_it or control.max_eval is too small, but 
                    may also be symptomatic of a badly scaled problem.

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

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- j_ne

		- is a scalar variable of type INT that holds the number of entries in $J$.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H_L$.

	*
		- c_l

		- is a one-dimensional array of size m and type T that holds the values $c_l$ of the lower bounds on the constraint functions $c(x)$. The j-th component of c_l, $i = 1, \ldots, m$, contains $c_{li}$.

	*
		- c_u

		- is a one-dimensional array of size m and type T that holds the values $c_u$ of the upper bounds on the constraint functionss $c(x)$. The j-th component of c_u, $i = 1, \ldots, m$, contains $c_{ui}$.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the values $x_l$ of the lower bounds on the optimization variables $x$. The j-th component of x_l, $j = 1, \ldots, n$, contains $x_{lj}$.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the values $x_u$ of the upper bounds on the optimization variables $x$. The j-th component of x_u, $j = 1, \ldots, n$, contains $x_{uj}$.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$. This should be set on input to an estimate of the minimizer.

	*
		- y

		- is a one-dimensional array of size m and type T that holds the values $y$ of the Lagrange multipliers. The j-th component of ``y``, i = 1, ... , m, contains $y_i$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the constraint functions $c(x)$. The i-th component of ``c``, i = 1, ... , m, contains $c_i(x)$.

	*
		- gl

		- is a one-dimensional array of size n and type T that holds the gradient $g_L(x,y,z) = \nabla_xf(x)$ of the Lagrangian function. The j-th component of ``gl``, j = 1, ... , n, contains $g_{Lj}$.

	*
		- eval_fc

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_c(n, x, f, c, userdata)

		  The value of the objective function $f(x)$ and 
                  the components of the residual function $c(x)$
		  evaluated at x=$x$ must be assigned to f and c, 
                  respectively, and the function return value set to 0. 
                  If the evaluation is impossible at x, return should 
                  be set to a nonzero value. Data may be passed into 
                  ``eval_fc`` via the structure ``userdata``.

	*
		- eval_gj

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_j(n, m, j_ne, x, g, j, userdata)

		  The components of the gradient $g = g(x)$ of the objective 
                  and Jacobian $J = \nabla_x c(x$) of the constraints 
                  evaluated at x=$x$ must be assigned to g and to j, 
                  in the same order as presented 
                  to expo_import, and the function return  value set to 0. 
                  If the evaluation is impossible at x,
		  return should be set to a nonzero value. Data may be
		  passed into ``eval_gj`` via the structure ``userdata``.

	*
		- eval_hl

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_hl(n, m, h_ne, x, y, h, userdata)

		  The nonzeros of the Hessian of the Lagrangian function
                  $H_L(x,y) = \nabla_{xx}f(x) -\sum_i y_i \nabla_{xx}c_i(x)$
		  evaluated at x=$x$ and y=$y$ must be assigned to h
		  in the same order as presented to expo_import, and the
		  function return value set to 0. If the evaluation is
		  impossible at x, return should be set to a nonzero
		  value. Data may be passed into ``eval_hl`` via the
		  structure ``userdata``.

.. index:: pair: function; expo_information
.. _doxid-galahad__expo_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; expo_terminate
.. _doxid-galahad__expo_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function expo_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`expo_control_type <doxid-structexpo__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`expo_inform_type <doxid-structexpo__inform__type>`)
