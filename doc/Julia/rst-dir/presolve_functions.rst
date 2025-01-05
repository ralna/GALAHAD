callable functions
------------------

.. index:: pair: function; presolve_initialize
.. _doxid-galahad__presolve_8h_1a30348a4e0a189046f55d995941693ed9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`presolve_control_type <doxid-structpresolve__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; presolve_read_specfile
.. _doxid-galahad__presolve_8h_1a78f57f6dd2885f41e9b79cc784ff673f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/presolve/PRESOLVE.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/presolve.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`presolve_control_type <doxid-structpresolve__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; presolve_import_problem
.. _doxid-galahad__presolve_8h_1aca96df1bce848a32af9f599a11c4c991:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_import_problem(T, INT, control, data, status, n, m, 
                                         H_type, H_ne, H_row, H_col, H_ptr, 
                                         H_val,  g, f, A_type, A_ne, A_row, 
                                         A_col, A_ptr, A_val, c_l, c_u, 
                                         x_l, x_u, n_out, m_out, H_ne_out,
                                         A_ne_out)

Import the initial data, and apply the presolve algorithm to report
crucial characteristics of the transformed variant

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`presolve_control_type <doxid-structpresolve__control__type>`)

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
                    'coordinate', 'sparse_by_rows' or 'diagonal' has
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
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- f

		- is a scalar of type T that holds the constant term $f$ of the objective function.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <details-ah_storage__unsym>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

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
		- n_out

		- is a scalar variable of type INT that holds the number of variables in the transformed problem.

	*
		- m_out

		- is a scalar variable of type INT that holds the number of general linear constraints in the transformed problem.

	*
		- H_ne_out

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H$ in the transformed problem.

	*
		- A_ne_out

		- is a scalar variable of type INT that holds the number of entries in $A$ in the transformed problem.

.. index:: pair: function; presolve_transform_problem
.. _doxid-galahad__presolve_8h_1af6da8ac04a1d4fdfd1b91cd8868791a1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_transform_problem(T, INT, data, status, n, m, H_ne, H_col, 
                                            H_ptr, H_val, g, f, A_ne, A_col, 
                                            A_ptr, A_val, c_l, c_u, x_l, x_u,
                                            y_l, y_u, z_l, z_u)

Apply the presolve algorithm to simplify the input problem, and output
the transformed variant

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

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
                    The input values n, m, A_ne or H_ne do not agree
                    with those output as necessary from
                    presolve_import_problem.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables in the transformed problem. This must match the value n_out from the last call to presolve_import_problem.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints. This must match the value m_out from the last call to presolve_import_problem.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the transformed $H$. This must match the value H_ne_out from the last call to presolve_import_problem.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type INT that holds the column indices of the lower triangular part of the transformed $H$ in the sparse row-wise storage scheme.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of the transformed $H$ in the sparse row-wise storage scheme.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the the transformed Hessian matrix $H$ in the sparse row-wise storage scheme.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the the transformed linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

	*
		- f

		- is a scalar of type T that holds the transformed constant term $f$ of the objective function.

	*
		- A_ne

		- is a scalar variable of type INT that holds the number of entries in the transformed $A$. This must match the value A_ne_out from the last call to presolve_import_problem.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type INT that holds the column indices of the transformed $A$ in the sparse row-wise storage scheme.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the transformed $A$, as well as the total number of entries, in the sparse row-wise storage scheme.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type T that holds the values of the entries of the transformed constraint Jacobian matrix $A$ in the sparse row-wise storage scheme.

	*
		- c_l

		- is a one-dimensional array of size m and type T that holds the transformed lower bounds $c^l$ on the constraints $A x$. The i-th component of ``c_l``, i = 1, ... , m, contains $c^l_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type T that holds the transformed upper bounds $c^l$ on the constraints $A x$. The i-th component of ``c_u``, i = 1, ... , m, contains $c^u_i$.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the transformed lower bounds $x^l$ on the variables $x$. The j-th component of ``x_l``, j = 1, ... , n, contains $x^l_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the transformed upper bounds $x^l$ on the variables $x$. The j-th component of ``x_u``, j = 1, ... , n, contains $x^l_j$.

	*
		- y_l

		- is a one-dimensional array of size m and type T that holds the implied lower bounds $y^l$ on the transformed Lagrange multipliers $y$. The i-th component of ``y_l``, i = 1, ... , m, contains $y^l_i$.

	*
		- y_u

		- is a one-dimensional array of size m and type T that holds the implied upper bounds $y^u$ on the transformed Lagrange multipliers $y$. The i-th component of ``y_u``, i = 1, ... , m, contains $y^u_i$.

	*
		- z_l

		- is a one-dimensional array of size m and type T that holds the implied lower bounds $y^l$ on the transformed dual variables $z$. The j-th component of ``z_l``, j = 1, ... , n, contains $z^l_i$.

	*
		- z_u

		- is a one-dimensional array of size m and type T that holds the implied upper bounds $y^u$ on the transformed dual variables $z$. The j-th component of ``z_u``, j = 1, ... , n, contains $z^u_i$.

.. index:: pair: function; presolve_restore_solution
.. _doxid-galahad__presolve_8h_1acf572e4805407de63003cd712f0fc495:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_restore_solution(T, INT, data, status, n_in, m_in, x_in, 
                                            c_in, y_in, z_in, n, m, x, c, y, z)

Given the solution (x_in,c_in,y_in,z_in) to the transformed problem, restore to recover the solution (x,c,y,z) to the original

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

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
                    The input values n, m, n_in and m_in do not agree
                    with those input to and output as necessary from
                    presolve_import_problem.

	*
		- n_in

		- is a scalar variable of type INT that holds the number of variables in the transformed problem. This must match the value n_out from the last call to presolve_import_problem.

	*
		- m_in

		- is a scalar variable of type INT that holds the number of general linear constraints. This must match the value m_out from the last call to presolve_import_problem.

	*
		- x_in

		- is a one-dimensional array of size n_in and type T that holds the transformed values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c_in

		- is a one-dimensional array of size m and type T that holds the transformed residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- y_in

		- is a one-dimensional array of size n_in and type T that holds the values $y$ of the transformed Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

	*
		- z_in

		- is a one-dimensional array of size n_in and type T that holds the values $z$ of the transformed dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables in the transformed problem. This must match the value n as input to presolve_import_problem.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints. This must match the value m as input to presolve_import_problem.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the transformed values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type T that holds the transformed residual $c(x)$. The i-th component of ``c``, j = 1, ... , m, contains $c_j(x)$.

	*
		- y

		- is a one-dimensional array of size n and type T that holds the values $y$ of the transformed Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

	*
		- z

		- is a one-dimensional array of size n and type T that holds the values $z$ of the transformed dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

.. index:: pair: function; presolve_information
.. _doxid-galahad__presolve_8h_1adc22ebe32d1361b83889645ff473ca9b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`presolve_inform_type <doxid-structpresolve__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; presolve_terminate
.. _doxid-galahad__presolve_8h_1abe2d3138390135885716064c3befb36b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function presolve_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`presolve_control_type <doxid-structpresolve__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`presolve_inform_type <doxid-structpresolve__inform__type>`)
