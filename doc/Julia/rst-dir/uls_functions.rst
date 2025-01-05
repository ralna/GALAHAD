callable functions
------------------

.. index:: pair: function; uls_initialize
.. _doxid-galahad__uls_8h_1a7afb5f2dde112e60686a5527a8f37ca4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_initialize(T, INT, solver, data, control, status)

Set default control values and initialize private data

Select solver, set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- solver

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`solver package <details-uls__solvers>` that should be used to factorize the matrix $A$. It should be one of 'gls', 'ma28', 'ma48 or 'getr'; lower or upper case variants are allowed.

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The initialization was successful.

		  * **-26**
                    The requested solver is not available.

.. index:: pair: function; uls_read_specfile
.. _doxid-galahad__uls_8h_1a5e2c9573bc8661114e9f073782b460ef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/uls/ULS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/uls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; uls_factorize_matrix
.. _doxid-galahad__uls_8h_1a6c0599479b84ee7d7c4ee7c473b76a83:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_factorize_matrix(T, INT, control, data, status, 
                                      m, n, type, ne, val, row, col, ptr)

Import matrix data into internal storage prior to solution, analyse the
sparsity patern, and subsequently factorize the matrix



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

		  Possible values are:

		  * **0**
                    The import, analysis and factorization were
                    conducted successfully.

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
                    The restrictions n > 0 and m> 0 or requirement that
                    the matrix type must contain the relevant string
                    'dense', 'coordinate' or 'sparse_by_rows has been
                    violated.

		  * **-26**
                    The requested solver is not available.

		  * **-29**
                    This option is not available with this solver.

		  * **-32**
                    More than control.max integer factor size words of
                    internal integer storage are required for in-core
                    factorization.

		  * **-50**
                    A solver-specific error occurred; check the
                    solver-specific information component of inform
                    along with the solver’s documentation for more
                    details.

	*
		- m

		- is a scalar variable of type INT that holds the number of rows in the unsymmetric matrix $A$.

	*
		- n

		- is a scalar variable of type INT that holds the number of columns in the unsymmetric matrix $A$.

	*
		- type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-a_storage__unsym>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- val

		- is a one-dimensional array of size ne and type T, that holds the values of the entries of the matrix $A$ in any of the supported storage schemes.

	*
		- row

		- is a one-dimensional array of size ne and type INT that holds the row indices of the matrix $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- col

		- is a one-dimensional array of size ne and type INT that holds the column indices of the matrix $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense storage schemes is used, and in this case can be C_NULL.

	*
		- ptr

		- is a one-dimensional array of size m+1 and type INT that holds the starting position of each row of the matrix $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; uls_reset_control
.. _doxid-galahad__uls_8h_1ad2ad6daa4d54d75e40fbe253f2bc5881:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; uls_solve_system
.. _doxid-galahad__uls_8h_1a01d3e7c19415125c660eba51d99c7518:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_solve_system(T, INT, data, status, m, n, sol, trans)

Solve the linear system $Ax=b$ or $A^Tx=b$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package.

		  Possible values are:

		  * **0**
                    The required solution was obtained.

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

		  * **-34**
                    The package PARDISO failed; check the
                    solver-specific information components
                    inform.pardiso iparm and inform.pardiso_dparm along
                    with PARDISO’s documentation for more details.

		  * **-35**
                    The package WSMP failed; check the solver-specific
                    information components inform.wsmp_iparm and
                    inform.wsmp dparm along with WSMP’s documentation
                    for more details.

	*
		- m

		- is a scalar variable of type INT that holds the number of rows in the unsymmetric matrix $A$.

	*
		- n

		- is a scalar variable of type INT that holds the number of columns in the unsymmetric matrix $A$.

	*
		- sol

		- is a one-dimensional array of size n and type T. On entry, it must hold the vector $b$. On a successful exit, its contains the solution $x$.

	*
		- trans

		- is a scalar variable of type Bool, that specifies whether to solve the equation $A^Tx=b$ (trans=true) or $Ax=b$ (trans=false).

.. index:: pair: function; uls_information
.. _doxid-galahad__uls_8h_1ab41cc4ccba208d7de3a0ccbc4b4efbcf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`uls_inform_type <doxid-structuls__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; uls_terminate
.. _doxid-galahad__uls_8h_1a36b2ea1ade2cdd8bca238f46e9e98435:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function uls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`uls_control_type <doxid-structuls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`uls_inform_type <doxid-structuls__inform__type>`)
