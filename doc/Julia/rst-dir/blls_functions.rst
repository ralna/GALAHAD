callable functions
------------------

.. index:: pair: function; blls_initialize
.. _doxid-galahad__blls_8h_1a12708c98f2473e03cd46f4dcfdb03409:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The import was successful.

.. index:: pair: function; blls_read_specfile
.. _doxid-galahad__blls_8h_1aa24c9c2fdaaaac84df5b98abbf84c859:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/blls/BLLS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/blls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; blls_import
.. _doxid-galahad__blls_8h_1afacd84f0b7592f4532cf7b77d278282f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_import(T, INT, control, data, status, n, o, 
                             Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are:

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
                    The restrictions n > 0, o > 0 or requirement that
                    type contains its relevant string 'coordinate',
                    'sparse_by_rows', 'sparse_by_columns',
                    'dense_by_rows', or 'dense_by_columns'; has been
                    violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- o

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- Ao_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-a_storage__unsym>` used for the Jacobian $A_o$. It should be one of 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense_by_rows', or 'dense_by_columns'; lower or upper case variants are allowed.

	*
		- Ao_ne

		- is a scalar variable of type INT that holds the number of entries in $A_o$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- Ao_row

		- is a one-dimensional array of size A_ne and type INT that holds the row indices of $A_o$ in the sparse co-ordinate or sparse column-wise storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- Ao_col

		- is a one-dimensional array of size A_ne and type INT that holds the column indices of $A_o$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- Ao_ptr_ne

		- is a scalar variable of type INT, that holds the length of the pointer array if sparse row or column storage scheme is used for $A_o$. For the sparse row scheme, Ao_ptr_ne should be at least o+1, while for the sparse column scheme, it should be at least n+1, It need not be set when the other schemes are used.

	*
		- Ao_ptr

		- is a one-dimensional array of size Ao_ptr_ne and type INT, that holds the starting position of each row of $A_o$, as well as the total number of entries, in the sparse row-wise storage scheme. By contrast, it holds the starting position of each column of $A_o$, as well as the total number of entries, in the sparse column-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; blls_import_without_a
.. _doxid-galahad__blls_8h_1a419f9b0769b4389beffbbc5f7d0fd58c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_import_without_a(T, INT, control, data, status, n, o)

Import problem data into internal storage prior to solution when $A_o$ is
not explicitly available.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

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
                    The restriction n > 0 or m > 0 has been violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- o

		- is a scalar variable of type INT that holds the number of residuals.

.. index:: pair: function; blls_reset_control
.. _doxid-galahad__blls_8h_1a96981ac9a0e3f44b2b38362fc3ab9991:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are:

		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; blls_solve_given_a
.. _doxid-galahad__blls_8h_1acf6d292989a5ac09f7f3e507283fb5bf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_solve_given_a(T, INT, data, userdata, status, n, o, 
                                    Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, 
                                    x_stat, w, eval_prec)

Solve the bound-constrained linear least-squares problem when the
Jacobian $A$ is available.


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
		  is a scalar variable of type INT that gives the entry and exit status from the package.

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
                    The restrictions n > 0, o > 0 or requirement that a
                    type contains its relevant string 'coordinate',
                    'sparse_by_rows', 'sparse_by_columns',
                    'dense_by_rows' or 'dense_by_columns' has been
                    violated.

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
		- o

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- Ao_ne

		- is a scalar variable of type INT that holds the number of entries in the design matrix $A_o$.

	*
		- Ao_val

		- is a one-dimensional array of size Ao_ne and type T that holds the values of the entries in the design matrix $A_o$ in any of the available storage schemes.

	*
		- b

		- is a one-dimensional array of size m and type T that holds the constant term $b$ in the residuals. The i-th component of ``b``, i = 1, ... , o, contains $b_i$.

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
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- r

		- is a one-dimensional array of size om and type T that holds the values of the residuals $r = A_o x - b$. The i-th component of ``r``, i = 1, ... , o, contains $r_i$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the values of the gradient $g = A_o^T W r$. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- w

		- is an optional one-dimensional array of size m and type T that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be C_NULL.

	*
		- eval_prec

		- is an optional user-supplied function that may be
		  C_NULL. If non-NULL, it must have the following
		  signature:

		  .. ref-code-block:: julia

		  	eval_prec(n, v, p, userdata)

		  The product $p = P^{-1} v$ involving the user's
		  preconditioner $P$ with the vector v=$v$, the result
		  $p$ must be retured in p, and the function return
		  value set to 0. If the evaluation is impossible,
		  return should be set to a nonzero value. Data may be
		  passed into ``eval_prec`` via the structure
		  ``userdata``.

.. index:: pair: function; blls_solve_reverse_a_prod
.. _doxid-galahad__blls_8h_1ac139bc1c65cf12cb532c4ab09f3af9d0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_solve_reverse_a_prod(T, INT, data, status, eval_status, n, o, b,
                                           x_l, x_u, x, z, r, g, x_stat, v, p,
                                           nz_v, nz_v_start, nz_v_end, nz_p, 
                                           nz_p_end, w)

Solve the bound-constrained linear least-squares problem when the
products of the Jacobian $A$ and its transpose with specified vectors
may be computed by the calling program.

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
                    The restriction n > 0 or requirement that a type
                    contains its relevant string 'coordinate',
                    'sparse_by_rows', 'sparse_by_columns',
                    'dense_by_rows' or 'dense_by_columns' has been
                    violated.

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

		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.

		  * **2**
                    The product $A_ov$ of the design matrix $A_o$ with a
                    given output vector $v$ is required from the
                    user. The vector $v$ will be stored in v and the
                    product $A_ov$ must be returned in p, status_eval
                    should be set to 0, and blls_solve_reverse_a_prod
                    re-entered with all other arguments unchanged. If
                    the product cannot be formed, v need not be set, but
                    blls_solve_reverse_a_prod should be re-entered with
                    eval_status set to a nonzero value.

		  * **3**
                    The product $A_o^Tv$ of the transpose of the residual
                    Jacobian $A_o$ with a given output vector $v$ is
                    required from the user. The vector $v$ will be
                    stored in v and the product $A_o^Tv$ must be returned
                    in p, status_eval should be set to 0, and
                    blls_solve_reverse_a_prod re-entered with all other
                    arguments unchanged. If the product cannot be
                    formed, v need not be set, but
                    blls_solve_reverse_a_prod should be re-entered with
                    eval_status set to a nonzero value.

		  * **4**
                    The product $A_ov$ of the design matrix $A_o$ with a
                    given sparse output vector $v$ is required from the
                    user. The nonzero components of the vector $v$ will
                    be stored as entries
                    nz_in[nz_in_start-1:nz_in_end-1] of v and the
                    product $A_ov$ must be returned in p, status_eval
                    should be set to 0, and blls_solve_reverse_a_prod
                    re-entered with all other arguments unchanged; The
                    remaining components of v should be ignored. If the
                    product cannot be formed, v need not be set, but
                    blls_solve_reverse_a_prod should be re-entered with
                    eval_status set to a nonzero value.

		  * **5**
                    The nonzero components of the product $A_ov$ of the
                    design matrix $A_o$ with a given sparse output
                    vector $v$ is required from the user. The nonzero
                    components of the vector $v$ will be stored as
                    entries nz_in[nz_in_start-1:nz_in_end-1] of v; the
                    remaining components of v should be ignored. The
                    resulting **nonzeros** in the product $A_ov$ must be
                    placed in their appropriate comnponents of p, while
                    a list of indices of the nonzeros placed in nz_out[0
                    : nz_out_end-1] and the number of nonzeros recorded
                    in nz_out_end. Additionally, status_eval should be
                    set to 0, and blls_solve_reverse_a_prod re-entered
                    with all other arguments unchanged. If the product
                    cannot be formed, v, nz_out_end and nz_out need not
                    be set, but blls_solve_reverse_a_prod should be
                    re-entered with eval_status set to a nonzero value.

		  * **6**
                    A subset of the product $A_o^Tv$ of the transpose of
                    the design matrix $A_o$ with a given output vector
                    $v$ is required from the user. The vector $v$ will
                    be stored in v and components
                    nz_in[nz_in_start-1:nz_in_end-1] of the product
                    $A_o^Tv$ must be returned in the relevant components
                    of p (the remaining components should not be set),
                    status_eval should be set to 0, and
                    blls_solve_reverse_a_prod re-entered with all other
                    arguments unchanged. If the product cannot be
                    formed, v need not be set, but
                    blls_solve_reverse_a_prod should be re-entered with
                    eval_status set to a nonzero value.

		  * **7**
                    The product $P^{-1}v$ of the inverse of the
                    preconditioner $P$ with a given output vector $v$ is
                    required from the user. The vector $v$ will be
                    stored in v and the product $P^{-1} v$ must be
                    returned in p, status_eval should be set to 0, and
                    blls_solve_reverse_a_prod re-entered with all other
                    arguments unchanged. If the product cannot be
                    formed, v need not be set, but
                    blls_solve_reverse_a_prod should be re-entered with
                    eval_status set to a nonzero value. This value of
                    status can only occur if the user has set
                    control.preconditioner = 2.

	*
		- eval_status

		- is a scalar variable of type INT that is used to indicate if the matrix products can be provided (see ``status`` above)

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- o

		- is a scalar variable of type INT that holds the number of residuals.

	*
		- b

		- is a one-dimensional array of size m and type T that holds the constant term $b$ in the residuals. The i-th component of ``b``, i = 1, ... , o, contains $b_i$.

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
		- r

		- is a one-dimensional array of size o and type T that holds the values of the residuals $r = A_o x - b$. The i-th component of ``r``, i = 1, ... , o, contains $r_i$.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the values of the gradient $g = A_o^T W r$. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type INT that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=2-4 above for details).

	*
		- p

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=2-4 above for details).

	*
		- nz_v

		- is a one-dimensional array of size n and type INT that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_v_start

		- is a scalar of type INT that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_v_end

		- is a scalar of type INT that is used for reverse communication (see status=3-4 above for details).

	*
		- nz_p

		- is a one-dimensional array of size n and type INT that is used for reverse communication (see status=4 above for details).

	*
		- nz_p_end

		- is a scalar of type INT that is used for reverse communication (see status=4 above for details).

	*
		- w

		- is an optional one-dimensional array of size o and type T that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be C_NULL.

.. index:: pair: function; blls_information
.. _doxid-galahad__blls_8h_1a457b8ee7c630715bcb43427f254b555f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`blls_inform_type <doxid-structblls__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; blls_terminate
.. _doxid-galahad__blls_8h_1ade863ffb6b142bfce669729f56911ac1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function blls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`blls_control_type <doxid-structblls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`blls_inform_type <doxid-structblls__inform__type>`)
