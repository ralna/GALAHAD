callable functions
------------------

.. index:: pair: function; nrek_initialize
.. _doxid-galahad__nrek_8h_1acb066d992c4ec394402bc7b7317e1163:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nrek_control_type <doxid-structnrek__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; nrek_read_specfile
.. _doxid-galahad__nrek_8h_1adc7c56e7be2f7cc9d32921582d379b13:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/nrek/NREK.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/nrek.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`nrek_control_type <doxid-structnrek__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; nrek_import
.. _doxid-galahad__nrek_8h_1a4becded30e9b95fe7028b7799292c0af:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_import(T, INT, control, data, status, n, 
                             H_type, H_ne, H_row, H_col, H_ptr)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`nrek_control_type <doxid-structnrek__control__type>`)

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
                    The restriction n > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', diagonal'
                    'scaled_identity', 'identity', 'zero' or 'none' 
                    has been violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows (and columns) of H.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-hs_storage__sym>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'; lower or upper case variants are allowed.

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

.. index:: pair: function; nrek_import_m
.. _doxid-galahad__nrek_8h_1a427420b6025d522bb7b3c652e8c2be48:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_import_s(T, INT, data, status, n, S_type, S_ne, S_row, S_col, S_ptr)

Import data for the scaling matrix $S$ into internal storage prior to solution.

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
                    The restriction n > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', diagonal' or
                    'coordinate', 'sparse_by_rows', diagonal',
                    'scaled_identity', or 'identity' has been violated.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows (and columns) of $S$.

	*
		- S_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-hs_storage__sym>` used for the scaling matrix, $S$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity' or 'identity'; lower or upper case variants are allowed.

	*
		- S_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- S_row

		- is a one-dimensional array of size S_ne and type INT that holds the row indices of the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- S_col

		- is a one-dimensional array of size S_ne and type INT that holds the column indices of the lower triangular part of $S$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or identity storage schemes are used, and in this case can be C_NULL.

	*
		- S_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $S$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; nrek_reset_control
.. _doxid-galahad__nrek_8h_1aae677e64bacb35354f49326815b694c3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`nrek_control_type <doxid-structnrek__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; nrek_solve_problem
.. _doxid-galahad__nrek_8h_1aadb8a751c29efcef663bf9560a1f9a8e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_solve_problem(T, INT, data, status, n, H_ne, H_val, c, 
                                    power, weight, x, S_ne, S_val)

Solve the norm-regularization subproblem.

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
                    The restrictions n > 0, power > 2 or weight > 0, or
                    requirement that a type contains its relevant string
                    'dense', 'coordinate', 'sparse_by_rows', 'diagonal'
                    'scaled_identity','identity', 'zero' or 'none' has 
                    been violated.

		  * **-9**
                    The analysis phase of the factorization of the
                    matrix $H$ or $S$ failed.

		  * **-10**
                    The factorization of the matrix $H$ or $S$ failed.

		  * **-11**
                    A solve involving the matrix $H$ or $S$ failed.

		  * **-15**
                    The matrix $S$ appears not to be diagonally dominant.

		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.

		  * **-18**
                    Too many iterations have been required. This may happen if
                    control.eks max is too small, but may also be symptomatic
                    of a badly scaled problem.

                  * **-31** 
                    A resolve call has been made before an initial call (see 
                    control.new_weight and control.new_values).

                  * **-38** 
                    An error occurred in a call to an LAPACK subroutine.
	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- H_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type T that holds the linear term $c$ of the objective function. The j-th component of ``c``, j = 1, ... , m, contains $c_j$.

	*
		- power

		- is a scalar of type T that holds the regularization power, $p$, used. power  must be strictly larger than two
	*
		- weight

		- is a scalar of type T that holds the regularization weight, $\Delta$, used. weight must be strictly positive

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- S_ne

		- is a scalar variable of type INT that holds the number of entries in the scaling matrix $S$ if it not the identity matrix.

	*
		- S_val

		- is a one-dimensional array of size S_ne and type T that holds the values of the entries of the scaling matrix $S$, if it is not the identity matrix, in any of the available storage schemes. If S_val is C_NULL, $S$ will be taken to be the identity matrix.

.. index:: pair: function; nrek_information
.. _doxid-galahad__nrek_8h_1a3dda24010e564e2d6536cc7ea518451e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`nrek_inform_type <doxid-structnrek__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; nrek_terminate
.. _doxid-galahad__nrek_8h_1ab5cf0077db0631814fdd03599a585376:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nrek_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nrek_control_type <doxid-structnrek__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`nrek_inform_type <doxid-structnrek__inform__type>`)
