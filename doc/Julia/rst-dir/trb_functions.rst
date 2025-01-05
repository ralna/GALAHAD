callable functions
------------------

.. index:: pair: function; trb_initialize
.. _doxid-galahad__trb_8h_1a9bffc46178a3e0b7eb2927d1c50440a1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see
                  :ref:`trb_control_type
                  <doxid-structtrb__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; trb_read_specfile
.. _doxid-galahad__trb_8h_1a4eaafdaf5187c8b91c119ce9395469e9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/trb/TRB.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/trb.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`trb_control_type <doxid-structtrb__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; trb_import
.. _doxid-galahad__trb_8h_1a13bc38fb28201adb78af7acf910ff0d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_import(T, INT, control, data, status, n, x_l, x_u,
                            H_type, ne, H_row, H_col, H_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`trb_control_type <doxid-structtrb__control__type>`)

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
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the values $x^l$ of the lower bounds on the optimization variables $x$. The j-th component of ``x_l``, $j = 1, \ldots, n$, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the values $x^u$ of the upper bounds on the optimization variables $x$. The j-th component of ``x_u``, $j = 1, \ldots, n$, contains $x^u_j$.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-hx_storage__sym>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to the Hessian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type INT that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type INT that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL

.. index:: pair: function; trb_reset_control
.. _doxid-galahad__trb_8h_1a550c3ca1966ea0fa9de84423b8658cd7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`trb_control_type <doxid-structtrb__control__type>`)

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

.. index:: pair: function; trb_solve_with_mat
.. _doxid-galahad__trb_8h_1a5a58e6c0c022eb451f14c82d653967f7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_solve_with_mat(T, INT, data, userdata, status, n, x, g, ne,
                                    eval_f, eval_g, eval_h, eval_prec)

Find a local minimizer of a given function subject to simple bounds on
the variables using a trust-region method.

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

		  * **-7**
                    The objective function appears to be unbounded from
                    below

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- eval_f

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_f(n, x, f, userdata)

		  The value of the objective function $f(x)$ evaluated
		  at x=$x$ must be assigned to f, and the function
		  return value set to 0. If the evaluation is impossible
		  at x, return should be set to a nonzero value. Data
		  may be passed into ``eval_f`` via the structure
		  ``userdata``.

	*
		- eval_g

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_g(n, x, g, userdata)

		  The components of the gradient $g = \nabla_x f(x$) of
		  the objective function evaluated at x=$x$ must be
		  assigned to g, and the function return value set
		  to 0. If the evaluation is impossible at x, return
		  should be set to a nonzero value. Data may be passed
		  into ``eval_g`` via the structure ``userdata``.

	*
		- eval_h

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_h(n, ne, x, h, userdata)

		  The nonzeros of the Hessian $H = \nabla_{xx}f(x)$ of
		  the objective function evaluated at x=$x$ must be
		  assigned to h in the same order as presented to
		  trb_import, and the function return value set to 0. If
		  the evaluation is impossible at x, return should be
		  set to a nonzero value. Data may be passed into
		  ``eval_h`` via the structure ``userdata``.

	*
		- eval_prec

		- is an optional user-supplied function that may be
		  C_NULL. If non-NULL, it must have the following
		  signature:

		  .. ref-code-block:: julia

		  	function eval_prec(n, x, u, v, userdata)

		  The product $u = P(x) v$ of the user's preconditioner
		  $P(x)$ evaluated at $x$ with the vector v=$v$, the
		  result $u$ must be retured in u, and the function
		  return value set to 0. If the evaluation is impossible
		  at x, return should be set to a nonzero value. Data
		  may be passed into ``eval_prec`` via the structure
		  ``userdata``.

.. index:: pair: function; trb_solve_without_mat
.. _doxid-galahad__trb_8h_1a376b81748cec0bf992542d80b8d38f49:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_solve_without_mat(T, INT, data, userdata, status, n, x, g,
                                       eval_f, eval_g, eval_hprod,
                                       eval_shprod, eval_prec)

Find a local minimizer of a given function subject to simple bounds on
the variables using a trust-region method.

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

		  * **-7**
                    The objective function appears to be unbounded from
                    below

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- eval_f

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_f(n, x, f, userdata)

		  The value of the objective function $f(x)$ evaluated
		  at x=$x$ must be assigned to f, and the function
		  return value set to 0. If the evaluation is impossible
		  at x, return should be set to a nonzero value. Data
		  may be passed into ``eval_f`` via the structure
		  ``userdata``.

	*
		- eval_g

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_g(n, x, g, userdata)

		  The components of the gradient $g = \nabla_x f(x$) of
		  the objective function evaluated at x=$x$ must be
		  assigned to g, and the function return value set
		  to 0. If the evaluation is impossible at x, return
		  should be set to a nonzero value. Data may be passed
		  into ``eval_g`` via the structure ``userdata``.

	*
		- eval_hprod

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_hprod(n, x, u, v, got_h, userdata)

		  The sum $u + \nabla_{xx}f(x) v$ of the product of the
		  Hessian $\nabla_{xx}f(x)$ of the objective function
		  evaluated at x=$x$ with the vector v=$v$ and the
		  vector $ $u$ must be returned in u, and the function
		  return value set to 0. If the evaluation is impossible
		  at x, return should be set to a nonzero value. The
		  Hessian has already been evaluated or used at x if
		  got_h is true. Data may be passed into ``eval_hprod``
		  via the structure ``userdata``.

	*
		- eval_shprod

		- is a user-supplied function that must have the
		  following signature:

		  .. ref-code-block:: julia

		  	function eval_shprod(n, x, nnz_v, index_nz_v, v, nnz_u,
                                 index_nz_u, u, got_h, userdata)

		  The product $u = \nabla_{xx}f(x) v$ of the Hessian
		  $\nabla_{xx}f(x)$ of the objective function evaluated
		  at $x$ with the sparse vector v=$v$ must be returned
		  in u, and the function return value set to 0. Only the
		  components index_nz_v[0:nnz_v-1] of v are nonzero, and
		  the remaining components may not have been be set. On
		  exit, the user must indicate the nnz_u indices of u
		  that are nonzero in index_nz_u[0:nnz_u-1], and only
		  these components of u need be set. If the evaluation
		  is impossible at x, return should be set to a nonzero
		  value. The Hessian has already been evaluated or used
		  at x if got_h is true. Data may be passed into
		  ``eval_prec`` via the structure ``userdata``.

	*
		- eval_prec

		- is an optional user-supplied function that may be
		  C_NULL. If non-NULL, it must have the following
		  signature:

		  .. ref-code-block:: julia


		  	function eval_prec(n, x, u, v, userdata)

		  The product $u = P(x) v$ of the user's preconditioner
		  $P(x)$ evaluated at $x$ with the vector v=$v$, the
		  result $u$ must be retured in u, and the function
		  return value set to 0. If the evaluation is impossible
		  at x, return should be set to a nonzero value. Data
		  may be passed into ``eval_prec`` via the structure
		  ``userdata``.

.. index:: pair: function; trb_solve_reverse_with_mat
.. _doxid-galahad__trb_8h_1a7bb520e36666386824b216a84be08837:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_solve_reverse_with_mat(T, INT, data, status, eval_status, n, x,
                                            f, g, ne, H_val, u, v)

Find a local minimizer of a given function subject to simple bounds on
the variables using a trust-region method.

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

		  * **-7**
                    The objective function appears to be unbounded from
                    below

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

		  * **2**
                    The user should compute the objective function value
                    $f(x)$ at the point $x$ indicated in x and then
                    re-enter the function. The required value should be
                    set in f, and eval_status should be set to 0. If the
                    user is unable to evaluate $f(x)$ for instance, if
                    the function is undefined at $x$ the user need not
                    set f, but should then set eval_status to a non-zero
                    value.

		  * **3**
                    The user should compute the gradient of the
                    objective function $\nabla_x f(x)$ at the point $x$
                    indicated in x and then re-enter the function. The
                    value of the i-th component of the gradient should
                    be set in g[i], for i = 1, ..., n and eval_status
                    should be set to 0. If the user is unable to
                    evaluate a component of $\nabla_x f(x)$ for instance
                    if a component of the gradient is undefined at
                    $x$ -the user need not set g, but should then set
                    eval_status to a non-zero value.

		  * **4**
                    The user should compute the Hessian of the objective
                    function $\nabla_{xx}f(x)$ at the point x indicated
                    in $x$ and then re-enter the function. The value
                    l-th component of the Hessian stored according to
                    the scheme input in the remainder of $H$ should be
                    set in H_val[l], for l = 0, ..., ne-1 and
                    eval_status should be set to 0. If the user is
                    unable to evaluate a component of $\nabla_{xx}f(x)$
                    for instance, if a component of the Hessian is
                    undefined at $x$ the user need not set H_val, but
                    should then set eval_status to a non-zero value.

		  * **6**
                    The user should compute the product $u = P(x)v$ of
                    their preconditioner $P(x)$ at the point x indicated
                    in $x$ with the vector $v$ and then re-enter the
                    function. The vector $v$ is given in v, the
                    resulting vector $u = P(x)v$ should be set in u and
                    eval_status should be set to 0. If the user is
                    unable to evaluate the product for instance, if a
                    component of the preconditioner is undefined at $x$
                    the user need not set u, but should then set
                    eval_status to a non-zero value.

	*
		- eval_status

		- is a scalar variable of type INT that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- f

		- is a scalar variable pointer of type T that holds the value of the objective function.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- u

		- is a one-dimensional array of size n and type T that is used for reverse communication (see above for details)

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see above for details)

.. index:: pair: function; trb_solve_reverse_without_mat
.. _doxid-galahad__trb_8h_1a95eac11acf02fe0d6eb4bc39ace5a100:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                               n, x, f, g, u, v, index_nz_v,
                                               nnz_v, index_nz_u, nnz_u)

Find a local minimizer of a given function subject to simple bounds on
the variables using a trust-region method.

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

		  * **-7**
                    The objective function appears to be unbounded from
                    below

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

		  * **-82**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

		  * **2**
                    The user should compute the objective function value
                    $f(x)$ at the point $x$ indicated in x and then
                    re-enter the function. The required value should be
                    set in f, and eval_status should be set to 0. If the
                    user is unable to evaluate $f(x)$ for instance, if
                    the function is undefined at $x$ the user need not
                    set f, but should then set eval_status to a non-zero
                    value.

		  * **3**
                    The user should compute the gradient of the
                    objective function $\nabla_x f(x)$ at the point $x$
                    indicated in x and then re-enter the function. The
                    value of the i-th component of the gradient should
                    be set in g[i], for i = 1, ..., n and eval_status
                    should be set to 0. If the user is unable to
                    evaluate a component of $\nabla_x f(x)$ for instance
                    if a component of the gradient is undefined at
                    $x$ -the user need not set g, but should then set
                    eval_status to a non-zero value.

		  * **5**
                    The user should compute the product
                    $\nabla_{xx}f(x)v$ of the Hessian of the objective
                    function $\nabla_{xx}f(x)$ at the point $x$
                    indicated in x with the vector $v$, add the result
                    to the vector $u$ and then re-enter the
                    function. The vectors $u$ and $v$ are given in u and
                    v respectively, the resulting vector $u +
                    \nabla_{xx}f(x)v$ should be set in u and eval_status
                    should be set to 0. If the user is unable to
                    evaluate the product for instance, if a component of
                    the Hessian is undefined at $x$ the user need not
                    alter u, but should then set eval_status to a
                    non-zero value.

		  * **6**
                    The user should compute the product $u = P(x)v$ of
                    their preconditioner $P(x)$ at the point x indicated
                    in $x$ with the vector $v$ and then re-enter the
                    function. The vector $v$ is given in v, the
                    resulting vector $u = P(x)v$ should be set in u and
                    eval_status should be set to 0. If the user is
                    unable to evaluate the product for instance, if a
                    component of the preconditioner is undefined at $x$
                    the user need not set u, but should then set
                    eval_status to a non-zero value.

		  * **7**
                    The user should compute the product $u =
                    \nabla_{xx}f(x)v$ of the Hessian of the objective
                    function $\nabla_{xx}f(x)$ at the point $x$
                    indicated in x with the **sparse** vector v=$v$ and
                    then re-enter the function. The nonzeros of $v$ are
                    stored in v[index_nz_v[0:nnz_v-1]] while the
                    nonzeros of $u$ should be returned in
                    u[index_nz_u[0:nnz_u-1]]; the user must set nnz_u
                    and index_nz_u accordingly, and set eval_status
                    to 0. If the user is unable to evaluate the product
                    for instance, if a component of the Hessian is
                    undefined at $x$ the user need not alter u, but
                    should then set eval_status to a non-zero value.

	*
		- eval_status

		- is a scalar variable of type INT that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- f

		- is a scalar variable pointer of type T that holds the value of the objective function.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- u

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=5,6,7 above for details)

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=5,6,7 above for details)

	*
		- index_nz_v

		- is a one-dimensional array of size n and type INT that is used for reverse communication (see status=7 above for details)

	*
		- nnz_v

		- is a scalar variable of type INT that is used for reverse communication (see status=7 above for details)

	*
		- index_nz_u

		- is a one-dimensional array of size n and type INT that is used for reverse communication (see status=7 above for details)

	*
		- nnz_u

		- is a scalar variable of type INT that is used for reverse communication (see status=7 above for details). On initial (status=1) entry, nnz_u should be set to an (arbitrary) nonzero value, and nnz_u=0 is recommended

.. index:: pair: function; trb_information
.. _doxid-galahad__trb_8h_1a105f41a31d49c59885c3372090bec776:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`trb_inform_type <doxid-structtrb__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit status from the package. Possible values are (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; trb_terminate
.. _doxid-galahad__trb_8h_1a739c7a44ddd0ce7b350cfa6da54948d0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function trb_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`trb_control_type <doxid-structtrb__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`trb_inform_type <doxid-structtrb__inform__type>`)
