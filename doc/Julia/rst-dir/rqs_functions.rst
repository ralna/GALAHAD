callable functions
------------------

.. index:: pair: function; rqs_initialize
.. _doxid-galahad__rqs_8h_1aeb8c3e1a278c83094aaaf185e9833fac:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`rqs_control_type <doxid-structrqs__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; rqs_read_specfile
.. _doxid-galahad__rqs_8h_1a1f6f3841ad5f7952dbc04a7cb19dd0e7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/rqs/RQS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/rqs.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`rqs_control_type <doxid-structrqs__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; rqs_import
.. _doxid-galahad__rqs_8h_1af815172e77293aa2a7c9dbcac2379f50:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_import(T, INT, control, data, status, n, 
                            H_type, H_ne, H_row, H_col, H_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`rqs_control_type <doxid-structrqs__control__type>`)

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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', diagonal' or
                    'identity' has been violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows (and columns) of H.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-ahm_storage__sym>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type INT that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type INT that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; rqs_import_m
.. _doxid-galahad__rqs_8h_1af0351d4956431c86e229f905041c222b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_import_m(T, INT, data, status, n, M_type, M_ne, M_row, M_col, M_ptr)

Import data for the scaling matrix M into internal storage prior to solution.

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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', diagonal' or
                    'identity' has been violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows (and columns) of M.

	*
		- M_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-ahm_storage__sym>` used for the scaling matrix, $M$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants are allowed.

	*
		- M_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $M$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- M_row

		- is a one-dimensional array of size M_ne and type INT that holds the row indices of the lower triangular part of $M$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- M_col

		- is a one-dimensional array of size M_ne and type INT that holds the column indices of the lower triangular part of $M$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or identity storage schemes are used, and in this case can be C_NULL.

	*
		- M_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $M$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; rqs_import_a
.. _doxid-galahad__rqs_8h_1a3d1116ac5c18fe085e902c77ec2776b5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_import_a(T, INT, data, status, m, A_type, A_ne, A_row, A_col, A_ptr)

Import data for the constraint matrix A into internal storage prior to solution.

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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate' or 'sparse_by_rows' has been violated.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints, i.e., the number of rows of A, if any. m must be non-negative.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-ahm_storage__unsym>` used for the constraint Jacobian, $A$ if any. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type INT that holds the number of entries in $A$, if used, in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type INT that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type INT that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; rqs_reset_control
.. _doxid-galahad__rqs_8h_1a86e1c32d2d07facbe602222e199a075f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`rqs_control_type <doxid-structrqs__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; rqs_solve_problem
.. _doxid-galahad__rqs_8h_1a162e2301c9d4bde7d57f5f1e820e2b84:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_solve_problem(T, INT, data, status, n, power, weight, f, c, 
                                  H_ne, H_val, x, M_ne, M_val, m, A_ne, A_val, y)

Solve the regularised quadratic problem.

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

		  On initial entry, status **must** be set to 1.

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
                    The restrictions n > 0, power > 2, weight > 0 and m
                    > 0 or requirement that a type contains its relevant
                    string 'dense', 'coordinate', 'sparse_by_rows',
                    'diagonal' or 'identity' has been violated.

		  * **-9**
                    The analysis phase of the factorization of the
                    matrix (1) failed.

		  * **-10**
                    The factorization of the matrix (1) failed.

		  * **-15**
                    The matrix M appears not to be diagonally dominant.

		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.

		  * **-18**
                    Too many factorizations have been required. This may
                    happen if control.max factorizations is too small,
                    but may also be symptomatic of a badly scaled
                    problem.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- power

		- is a scalar of type T that holds the order of regularisation, $p$, used. power must be no smaller than 2.

	*
		- weight

		- is a scalar of type T that holds the regularisation weight, $\sigma$, used. weight must be strictly positive.

	*
		- c

		- is a one-dimensional array of size n and type T that holds the linear term $c$ of the objective function. The j-th component of ``c``, j = 1, ... , m, contains $c_j$.

	*
		- f

		- is a scalar of type T that holds the constant term $f$ of the objective function.

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- M_ne

		- is a scalar variable of type INT that holds the number of entries in the scaling matrix $M$ if it not the identity matrix.

	*
		- M_val

		- is a one-dimensional array of size M_ne and type T that holds the values of the entries of the scaling matrix $M$, if it is not the identity matrix, in any of the available storage schemes. If M_val is C_NULL, M will be taken to be the identity matrix.

	*
		- m

		- is a scalar variable of type INT that holds the number of general linear constraints, if any. m must be non-negative.

	*
		- A_ne

		- is a scalar variable of type INT that holds the number of entries in the constraint Jacobian matrix $A$ if used. A_ne must be non-negative.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type T that holds the values of the entries of the constraint Jacobian matrix $A$, if used, in any of the available storage schemes. If A_val is C_NULL, no constraints will be enforced.

	*
		- y

		- is a one-dimensional array of size m and type T that holds the values $y$ of the Lagrange multipliers for the equality constraints $A x = 0$ if used. The i-th component of ``y``, i = 1, ... , m, contains $y_i$.

.. index:: pair: function; rqs_information
.. _doxid-galahad__rqs_8h_1a586e85ec11c4647346916f49805fcb83:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`rqs_inform_type <doxid-structrqs__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; rqs_terminate
.. _doxid-galahad__rqs_8h_1ae1f727eadfaada300dc6a5e268ac2b74:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rqs_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`rqs_control_type <doxid-structrqs__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`rqs_inform_type <doxid-structrqs__inform__type>`)
