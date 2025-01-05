.. _global:

callable functions
------------------

.. index:: pair: function; nls_initialize
.. _doxid-galahad__nls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_initialize(T, INT, data, control, inform)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

.. index:: pair: function; nls_read_specfile
.. _doxid-galahad__nls_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/nls/NLS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/nls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; nls_import
.. _doxid-galahad__nls_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_import(T, INT, control, data, status, n, m,
                            J_type, J_ne, J_row, J_col, J_ptr,
                            H_type, H_ne, H_row, H_col, H_ptr,
                            P_type, P_ne, P_row, P_col, P_ptr, w)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

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
                    J/H/P_type contains its relevant string 'dense',
                    'dense_by_columns', 'coordinate', 'sparse_by_rows',
                    'sparse_by_columns', 'diagonal' or 'absent' has been
                    violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- J_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-nls_storage__unsym>` used for the Jacobian, $J$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

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

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <details-nls_storage__sym>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to $H$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type INT that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type INT that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

	*
		- P_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <details-nls_storage__unsym>` used for the residual-Hessians-vector product matrix, $P$. It should be one of 'coordinate', 'sparse_by_columns', 'dense_by_columns' or 'absent', the latter if access to $P$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- P_ne

		- is a scalar variable of type INT that holds the number of entries in $P$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- P_row

		- is a one-dimensional array of size P_ne and type INT that holds the row indices of $P$ in either the sparse co-ordinate, or the sparse column-wise storage scheme. It need not be set when the dense storage scheme is used, and in this case can be C_NULL.

	*
		- P_col

		- is a one-dimensional array of size P_ne and type INT that holds the row indices of $P$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- P_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $P$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

	*
		- w

		- is a one-dimensional array of size m and type T that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be C_NULL.

.. index:: pair: function; nls_reset_control
.. _doxid-galahad__nls_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

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

.. index:: pair: function; nls_solve_with_mat
.. _doxid-galahad__nls_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_solve_with_mat(T, INT, data, userdata, status, n, m, x, c, g,
                                    eval_c, j_ne, eval_j, h_ne, eval_h,
                                    p_ne, eval_hprods)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where $H = \nabla_{xx}f(x)$ is provided
specifically, and all function/derivative information is available by
function calls.

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
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

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
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- eval_c

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_c(n, x, c, userdata)

		  The componnts of the residual function $c(x)$
		  evaluated at x=$x$ must be assigned to c, and the
		  function return value set to 0. If the evaluation is
		  impossible at x, return should be set to a nonzero
		  value. Data may be passed into ``eval_c`` via the
		  structure ``userdata``.

	*
		- j_ne

		- is a scalar variable of type INT that holds the number of entries in the Jacobian matrix $J$.

	*
		- eval_j

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_j(n, m, jne, x, j, userdata)

		  The components of the Jacobian $J = \nabla_x c(x$) of
		  the residuals must be assigned to j in the same order
		  as presented to nls_import, and the function return
		  value set to 0. If the evaluation is impossible at x,
		  return should be set to a nonzero value. Data may be
		  passed into ``eval_j`` via the structure ``userdata``.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$ if it is used.

	*
		- eval_h

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_h(n, m, hne, x, y, h, userdata)

		  The nonzeros of the matrix $H = \sum_{i=1}^m y_i
		  \nabla_{xx}c_i(x)$ of the weighted residual Hessian
		  evaluated at x=$x$ and y=$y$ must be assigned to h
		  in the same order as presented to nls_import, and the
		  function return value set to 0. If the evaluation is
		  impossible at x, return should be set to a nonzero
		  value. Data may be passed into ``eval_h`` via the
		  structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type INT that holds the number of entries in the residual-Hessians-vector product matrix $P$ if it is used.

	*
		- eval_hprods

		- is an optional user-supplied function that may be
		  C_NULL. If non-NULL, it must have the following
		  signature:

		  .. ref-code-block:: julia

		  	function eval_hprods(n, m, pne, x, v, p, got_h, userdata)

		  The entries of the matrix $P$, whose i-th column is
		  the product $\nabla_{xx}c_i(x) v$ between
		  $\nabla_{xx}c_i(x)$, the Hessian of the i-th component
		  of the residual $c(x)$ at x=$x$, and v=$v$ must be
		  returned in p and the function return value set
		  to 0. If the evaluation is impossible at x, return
		  should be set to a nonzero value. Data may be passed
		  into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_without_mat
.. _doxid-galahad__nls_8h_1a692ecbfaa428584e60aa4c33d7278a64:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_solve_without_mat(T, INT, data, userdata, status, n, m, x, c, g,
                                       eval_c, eval_jprod, eval_hprod,
                                       p_ne, eval_hprods)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to $H = \nabla_{xx}f(x)$ is
provided by Hessian-vector products, and all function/derivative
information is available by function calls.



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
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- eval_c

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_c(n, x, c, userdata)

		  The componnts of the residual function $c(x)$
		  evaluated at x=$x$ must be assigned to c, and the
		  function return value set to 0. If the evaluation is
		  impossible at x, return should be set to a nonzero
		  value. Data may be passed into ``eval_c`` via the
		  structure ``userdata``.

	*
		- eval_jprod

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_jprod(n, m, x, transpose, u, v, got_j, userdata)

		  The sum $u + \nabla_{x}c_(x) v$ (if the Bool transpose
		  is false) or The sum $u + (\nabla_{x}c_(x))^T v$ (if
		  tranpose is true) bewteen the product of the Jacobian
		  $\nabla_{x}c_(x)$ or its tranpose with the vector
		  v=$v$ and the vector $ $u$ must be returned in u, and
		  the function return value set to 0. If the evaluation
		  is impossible at x, return should be set to a nonzero
		  value. Data may be passed into ``eval_jprod`` via the
		  structure ``userdata``.

	*
		- eval_hprod

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_hprod(n, m, x, y, u, v, got_h, userdata)

		  The sum $u + \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v$ of
		  the product of the weighted residual Hessian $H =
		  \sum_{i=1}^m y_i \nabla_{xx}c_i(x)$ evaluated at
		  x=$x$ and y=$y$ with the vector v=$v$ and the vector
		  $u$ must be returned in u, and the function return
		  value set to 0. If the evaluation is impossible at x,
		  return should be set to a nonzero value. The Hessians
		  have already been evaluated or used at x if the Bool
		  got_h is true. Data may be passed into ``eval_hprod``
		  via the structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type INT that holds the number of entries in the residual-Hessians-vector product matrix $P$ if it is used.

	*
		- eval_hprods

		- is an optional user-supplied function that may be
		  C_NULL. If non-NULL, it must have the following
		  signature:

		  .. ref-code-block:: julia

		  	function eval_hprods(n, m, p_ne, x, v, pval, got_h, userdata)

		  The entries of the matrix $P$, whose i-th column is
		  the product $\nabla_{xx}c_i(x) v$ between
		  $\nabla_{xx}c_i(x)$, the Hessian of the i-th component
		  of the residual $c(x)$ at x=$x$, and v=$v$ must be
		  returned in pval and the function return value set
		  to 0. If the evaluation is impossible at x, return
		  should be set to a nonzero value. Data may be passed
		  into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_reverse_with_mat
.. _doxid-galahad__nls_8h_1a9ad89605640c53c33ddd5894b5e3edd1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                            n, m, x, c, g, j_ne, J_val,
                                            y, h_ne, H_val, v, p_ne, P_val)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where $H = \nabla_{xx}f(x)$ is provided
specifically, but function/derivative information is only available by
returning to the calling procedure

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
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

		  * **2**
                    The user should compute the vector of residuals
                    $c(x)$ at the point $x$ indicated in x and then
                    re-enter the function. The required value should be
                    set in c, and eval_status should be set to 0. If the
                    user is unable to evaluate $c(x)$ for instance, if
                    the function is undefined at $x$ the user need not
                    set c, but should then set eval_status to a non-zero
                    value.

		  * **3**
                    The user should compute the Jacobian of the vector
                    of residual functions, $\nabla_x c(x)$, at the point
                    $x$ indicated in x and then re-enter the
                    function. The l-th component of the Jacobian stored
                    according to the scheme specified for the remainder
                    of $J$ in the earlier call to nls_import should be
                    set in J_val[l], for l = 0, ..., J_ne-1 and
                    eval_status should be set to 0. If the user is
                    unable to evaluate a component of $J$ for instance,
                    if a component of the matrix is undefined at $x$ the
                    user need not set J_val, but should then set
                    eval_status to a non-zero value.

		  * **4**
                    The user should compute the matrix $H = \sum_{i=1}^m
                    v_i \nabla_{xx}c_i(x)$ of weighted residual Hessian
                    evaluated at x=$x$ and v=$v$ and then re-enter the
                    function. The l-th component of the matrix stored
                    according to the scheme specified for the remainder
                    of $H$ in the earlier call to nls_import should be
                    set in H_val[l], for l = 0, ..., H_ne-1 and
                    eval_status should be set to 0. If the user is
                    unable to evaluate a component of $H$ for instance,
                    if a component of the matrix is undefined at $x$ the
                    user need not set H_val, but should then set
                    eval_status to a non-zero value. ****Note** that this
                    return will not happen if the Gauss-Newton model is
                    selected**

		  * **7**
                    The user should compute the entries of the matrix
                    $P$, whose i-th column is the product
                    $\nabla_{xx}c_i(x) v$ between $\nabla_{xx}c_i(x)$,
                    the Hessian of the i-th component of the residual
                    $c(x)$ at x=$x$, and v=$v$ and then re-enter the
                    function. The l-th component of the matrix stored
                    according to the scheme specified for the remainder
                    of $P$ in the earlier call to nls_import should be
                    set in P_val[l], for l = 0, ..., P_ne-1 and
                    eval_status should be set to 0. If the user is
                    unable to evaluate a component of $P$ for instance,
                    if a component of the matrix is undefined at $x$ the
                    user need not set P_val, but should then set
                    eval_status to a non-zero value. **Note** that this
                    return will not happen if either the Gauss-Newton or
                    Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type INT that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- j_ne

		- is a scalar variable of type INT that holds the number of entries in the Jacobian matrix $J$.

	*
		- J_val

		- is a one-dimensional array of size j_ne and type T that holds the values of the entries of the Jacobian matrix $J$ in any of the available storage schemes. See status = 3, above, for more details.

	*
		- y

		- is a one-dimensional array of size m and type T that is used for reverse communication. See status = 4 above for more details.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes. See status = 4, above, for more details.

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication. See status = 7, above, for more details.

	*
		- p_ne

		- is a scalar variable of type INT that holds the number of entries in the residual-Hessians-vector product matrix, $P$.

	*
		- P_val

		- is a one-dimensional array of size p_ne and type T that holds the values of the entries of the residual-Hessians-vector product matrix, $P$. See status = 7, above, for more details.

.. index:: pair: function; nls_solve_reverse_without_mat
.. _doxid-galahad__nls_8h_1a6dddd928c19adec0abf76bdb2d75da17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                               n, m, x, c, g, transpose,
                                               u, v, y, p_ne, P_val)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to $H = \nabla_{xx}f(x)$ is
provided by Hessian-vector products, but function/derivative information
is only available by returning to the calling procedure.

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
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

		  * **2**
                    The user should compute the vector of residuals
                    $c(x)$ at the point $x$ indicated in x and then
                    re-enter the function. The required value should be
                    set in c, and eval_status should be set to 0. If the
                    user is unable to evaluate $c(x)$ for instance, if
                    the function is undefined at $x$ the user need not
                    set c, but should then set eval_status to a non-zero
                    value.

		  * **5**
                    The user should compute the sum $u + \nabla_{x}c_(x)
                    v$ (if tranpose is false) or $u +
                    (\nabla_{x}c_(x))^T v$ (if tranpose is true) between
                    the product of the Jacobian $\nabla_{x}c_(x)$ or its
                    tranpose with the vector v=$v$ and the vector 
                    u=$u$, and then re-enter the function. The result
                    should be set in u, and eval_status should be set
                    to 0. If the user is unable to evaluate the sum for
                    instance, if the Jacobian is undefined at $x$ the
                    user need not set u, but should then set eval_status
                    to a non-zero value.

		  * **6**
                    The user should compute the sum $u + \sum_{i=1}^m
                    y_i \nabla_{xx}c_i(x) v$ between the product of the
                    weighted residual Hessian $H = \sum_{i=1}^m y_i
                    \nabla_{xx}c_i(x)$ evaluated at x=$x$ and y=$y$
                    with the vector v=$v$ and the the vector u=$u$,
                    and then re-enter the function. The result should be
                    set in u, and eval_status should be set to 0. If the
                    user is unable to evaluate the sum for instance, if
                    the weifghted residual Hessian is undefined at $x$
                    the user need not set u, but should then set
                    eval_status to a non-zero value.

		  * **7**
                    The user should compute the entries of the matrix
                    $P$, whose i-th column is the product
                    $\nabla_{xx}c_i(x) v$ between $\nabla_{xx}c_i(x)$,
                    the Hessian of the i-th component of the residual
                    $c(x)$ at x=$x$, and v=$v$ and then re-enter the
                    function. The l-th component of the matrix stored
                    according to the scheme specified for the remainder
                    of $P$ in the earlier call to nls_import should be
                    set in P_val[l], for l = 0, ..., P_ne-1 and
                    eval_status should be set to 0. If the user is
                    unable to evaluate a component of $P$ for instance,
                    if a component of the matrix is undefined at $x$ the
                    user need not set P_val, but should then set
                    eval_status to a non-zero value. **Note** that this
                    return will not happen if either the Gauss-Newton or
                    Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type INT that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- m

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- transpose

		- is a scalar variable of type Bool, that indicates whether the product with Jacobian or its transpose should be obtained when status=5.

	*
		- u

		- is a one-dimensional array of size max(n,m) and type T that is used for reverse communication. See status = 5,6 above for more details.

	*
		- v

		- is a one-dimensional array of size max(n,m) and type T that is used for reverse communication. See status = 5,6,7 above for more details.

	*
		- y

		- is a one-dimensional array of size m and type T that is used for reverse communication. See status = 6 above for more details.

	*
		- p_ne

		- is a scalar variable of type INT that holds the number of entries in the residual-Hessians-vector product matrix, $P$.

	*
		- P_val

		- is a one-dimensional array of size P_ne and type T that holds the values of the entries of the residual-Hessians-vector product matrix, $P$. See status = 7, above, for more details.

.. index:: pair: function; nls_information
.. _doxid-galahad__nls_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; nls_terminate
.. _doxid-galahad__nls_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)
