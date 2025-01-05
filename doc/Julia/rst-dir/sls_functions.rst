callable functions
------------------

.. index:: pair: function; sls_initialize
.. _doxid-galahad__sls_8h_1a1d8a0c73587ca6d7f5333d41b3e2472a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_initialize(T, INT, solver, data, control, status)

Select solver, set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- solver

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`solver package<details-sls__solvers>` that should be used to factorize the matrix $A$. It should be one of 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl pardiso', 'pastix', 'wsmp', 'potr', 'sytr' or 'pbtr'; lower or upper case variants are allowed.

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sls_control_type <doxid-structsls__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The initialization was successful.

		  * **-26**
                    The requested solver is not available.

.. index:: pair: function; sls_read_specfile
.. _doxid-galahad__sls_8h_1ada1e7b9ed799335702f85a551b64bf88:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/sls/SLS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/sls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`sls_control_type <doxid-structsls__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; sls_analyse_matrix
.. _doxid-galahad__sls_8h_1a380a7f50cc71c705d15a791acde946cf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_analyse_matrix(T, INT, control, data, status, n, type, ne, row, col, ptr)

Import structural matrix data into internal storage prior to solution

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sls_control_type <doxid-structsls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

		  Possible values are:

		  * **0**
                    The import and analysis were conducted successfully.

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
                    The restrictions n > 0 or requirement that the
                    matrix type must contain the relevant string
                    'dense', 'coordinate' or 'sparse_by_rows has been
                    violated.

		  * **-20**
                    The matrix is not positive definite while the solver
                    used expected it to be.

		  * **-26**
                    The requested solver is not available.

		  * **-29**
                    This option is not available with this solver.

		  * **-32**
                    More than control.max integer factor size words of
                    internal integer storage are required for in-core
                    factorization.

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

		  * **-36**
                    The scaling package HSL MC64 failed; check the
                    solver-specific information component
                    inform.mc64_info along with HSL MC64’s documentation
                    for more details.

		  * **-37**
                    The scaling package MC77 failed; check the
                    solver-specific information components inform.mc77
                    info and inform.mc77_rinfo along with MC77’s
                    documentation for more details.

		  * **-43**
                    A direct-access file error occurred. See the value
                    of inform.ma77_info.flag for more details.

		  * **-50**
                    A solver-specific error occurred; check the
                    solver-specific information component of inform
                    along with the solver’s documentation for more
                    details.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows in the symmetric matrix $A$.

	*
		- type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-s_storage__sym>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- row

		- is a one-dimensional array of size ne and type INT that holds the row indices of the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- col

		- is a one-dimensional array of size ne and type INT that holds the column indices of the lower triangular part of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense storage scheme is used, and in this case can be C_NULL.

	*
		- ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; sls_reset_control
.. _doxid-galahad__sls_8h_1aacc344b8cdf0b1c27965f191382372e4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sls_control_type <doxid-structsls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit status from the package. Possible values are:

		  * 0. The import was successful.

.. index:: pair: function; sls_factorize_matrix
.. _doxid-galahad__sls_8h_1ab6666f5eb7b0bdbbc9c9b52b7a2e2c41:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_factorize_matrix(T, INT, data, status, ne, val)

Form and factorize the symmetric matrix $A$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

		  Possible values are:

		  * **0**
                    The factors were generated successfully.

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
                    The restrictions n > 0 or requirement that the
                    matrix type must contain the relevant string
                    'dense', 'coordinate' or 'sparse_by_rows has been
                    violated.

		  * **-20**
                    The matrix is not positive definite while the solver
                    used expected it to be.

		  * **-26**
                    The requested solver is not available.

		  * **-29**
                    This option is not available with this solver.

		  * **-32**
                    More than control.max integer factor size words of
                    internal integer storage are required for in-core
                    factorization.

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

		  * **-36**
                    The scaling package HSL MC64 failed; check the
                    solver-specific information component
                    inform.mc64_info along with HSL MC64’s documentation
                    for more details.

		  * **-37**
                    The scaling package MC77 failed; check the
                    solver-specific information components inform.mc77
                    info and inform.mc77_rinfo along with MC77’s
                    documentation for more details.

		  * **-43**
                    A direct-access file error occurred. See the value
                    of inform.ma77_info.flag for more details.

		  * **-50**
                    A solver-specific error occurred; check the
                    solver-specific information component of inform
                    along with the solver’s documentation for more
                    details.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the symmetric matrix $A$.

	*
		- val

		- is a one-dimensional array of size ne and type T that holds the values of the entries of the lower triangular part of the symmetric matrix $A$ in any of the supported storage schemes.

.. index:: pair: function; sls_solve_system
.. _doxid-galahad__sls_8h_1a1b3e7546b59b06160c51e16b6781bc0b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_solve_system(T, INT, data, status, n, sol)

Solve the linear system $Ax=b$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

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
		- n

		- is a scalar variable of type INT that holds the number of entries in the vectors $b$ and $x$.

	*
		- sol

		- is a one-dimensional array of size n and type double. On entry, it must hold the vector $b$. On a successful exit, its contains the solution $x$.

.. index:: pair: function; sls_partial_solve_system
.. _doxid-galahad__sls_8h_1ac66dc50d8b54acab90d70ae649b92905:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_partial_solve_system(T, INT, part, data, status, n, sol)

Given the factorization $A = L D U$ with $U = L^T$, solve the linear
system $$Mx=b,$$ where $M$ is one of $L$, $D$, $U$ or $S = L \sqrt{D}$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- part

		- is a one-dimensional array of type Vararg{Cchar} that specifies the component $M$ of the factorization that is to be used. It should be one of "L", "D", "U" or "S", and these correspond to the parts $L$, $D$, $U$ and $S$; lower or upper case variants are allowed.

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
		- n

		- is a scalar variable of type INT that holds the number of entries in the vectors $b$ and $x$.

	*
		- sol

		- is a one-dimensional array of size n and type double. On entry, it must hold the vector $b$. On a successful exit, its contains the solution $x$.

.. index:: pair: function; sls_information
.. _doxid-galahad__sls_8h_1a0ca4a126813c3aafac9d791a152b233c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_information(T, INT, data, inform, status)

Provide output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`sls_inform_type <doxid-structsls__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sls_terminate
.. _doxid-galahad__sls_8h_1aa5aafa378e3500ce31783e13c3395d30:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sls_control_type <doxid-structsls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`sls_inform_type <doxid-structsls__inform__type>`)
