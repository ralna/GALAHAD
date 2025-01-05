callable functions
------------------

.. index:: pair: function; sbls_initialize
.. _doxid-galahad__sbls_8h_1a30b1a9463e4abd5cfa0150ffb30569a9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; sbls_read_specfile
.. _doxid-galahad__sbls_8h_1abde2e76567a4c8721fe9c2386106e972:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/sbls/SBLS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/sbls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; sbls_import
.. _doxid-galahad__sbls_8h_1ab7cbabccf52f8be7ae417e089eba4b82:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_import(T, INT, control, data, status, n, m, 
                             H_type, H_ne, H_row, H_col, H_ptr, 
                             A_type, A_ne, A_row, A_col, A_ptr, 
                             C_type, C_ne, C_row, C_col, C_ptr)

Import structural matrix data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

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

	*
		- n

		- is a scalar variable of type INT that holds the number of rows in the symmetric matrix $H$.

	*
		- m

		- is a scalar variable of type INT that holds the number of rows in the symmetric matrix $C$.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-sbls_storage__sym>` used for the matrix $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if $H=0$; lower or upper case variants are allowed.

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

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <details-sbls_storage__unsym>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

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
		- C_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <details-sbls_storage__sym>` used for the matrix $C$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if $C=0$; lower or upper case variants are allowed.

	*
		- C_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $C$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- C_row

		- is a one-dimensional array of size C_ne and type INT that holds the row indices of the lower triangular part of $C$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be C_NULL.

	*
		- C_col

		- is a one-dimensional array of size C_ne and type INT that holds the column indices of the lower triangular part of $C$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be C_NULL.

	*
		- C_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $C$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

.. index:: pair: function; sbls_reset_control
.. _doxid-galahad__sbls_8h_1afdfe80ab659c2936d23802b6a6103eb8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The import was successful.

.. index:: pair: function; sbls_factorize_matrix
.. _doxid-galahad__sbls_8h_1a77799da1282c3567b56ae8db42b75f65:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_factorize_matrix(T, INT, data, status, n, h_ne, H_val, 
	                               a_ne, A_val, c_ne, C_val, D)

Form and factorize the block matrix
$$K_{G} = \begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix}$$
for some appropriate matrix $G$.

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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.

		  * **-9**
                    An error was reported by SLS analyse. The return
                    status from SLS analyse is given in
                    inform.sls_inform.status. See the documentation for
                    the GALAHAD package SLS for further details.

		  * **-10**
                    An error was reported by SLS_factorize. The return
                    status from SLS factorize is given in
                    inform.sls_inform.status. See the documentation for
                    the GALAHAD package SLS for further details.

		  * **-13**
                    An error was reported by ULS_factorize. The return
                    status from ULS_factorize is given in
                    inform.uls_factorize_status. See the documentation
                    for the GALAHAD package ULS for further details.

		  * **-15**
                    The computed preconditioner $K_G$ is singular and is
                    thus unsuitable.

		  * **-20**
                    The computed preconditioner $K_G$ has the wrong
                    inertia and is thus unsuitable.

		  * **-24**
                    An error was reported by the GALAHAD package
                    SORT_reorder_by_rows. The return status from
                    SORT_reorder_by_rows is given in
                    inform.sort_status. See the documentation for the
                    GALAHAD package SORT for further details.

	*
		- n

		- is a scalar variable of type INT that holds the number of rows in the symmetric matrix $H$.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the symmetric matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the symmetric matrix $H$ in any of the available storage schemes

	*
		- a_ne

		- is a scalar variable of type INT that holds the number of entries in the unsymmetric matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type T that holds the values of the entries of the unsymmetric matrix $A$ in any of the available storage schemes.

	*
		- c_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the symmetric matrix $C$.

	*
		- C_val

		- is a one-dimensional array of size c_ne and type T that holds the values of the entries of the lower triangular part of the symmetric matrix $C$ in any of the available storage schemes

	*
		- D

		- is a one-dimensional array of size n and type T that holds the values of the entries of the diagonal matrix $D$ that is required if the user has specified control.preconditioner = 5. It need not be set otherwise.

.. index:: pair: function; sbls_solve_system
.. _doxid-galahad__sbls_8h_1a2c3ae7b15fc1c43771d395540c37b9fa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_solve_system(T, INT, data, status, n, m, sol)

Solve the block linear system
$$\begin{pmatrix}G & A^T \\ A  & - C\end{pmatrix} 
\begin{pmatrix}x \\ y\end{pmatrix} = 
\begin{pmatrix}a \\ b\end{pmatrix}.$$

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

		  * **-11**
                    An error was reported by SLS_solve. The return
                    status from SLS solve is given in
                    inform.sls_inform.status. See the documentation for
                    the GALAHAD package SLS for further details.

		  * **-14**
                    An error was reported by ULS_solve. The return
                    status from ULS_solve is given in
                    inform.uls_solve_status. See the documentation for
                    the GALAHAD package ULS for further details.

	*
		- n

		- is a scalar variable of type INT that holds the number of entries in the vector $a$.

	*
		- m

		- is a scalar variable of type INT that holds the number of entries in the vector $b$.

	*
		- sol

		- is a one-dimensional array of size n + m and type T. on entry, its first n entries must hold the vector $a$, and the following entries must hold the vector $b$. On a successful exit, its first n entries contain the solution components $x$, and the following entries contain the components $y$.

.. index:: pair: function; sbls_information
.. _doxid-galahad__sbls_8h_1a9f93f5c87ae0088ceb72c4f7e73c9418:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`sbls_inform_type <doxid-structsbls__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sbls_terminate
.. _doxid-galahad__sbls_8h_1a73d7d29d113a62c48cc176146539bca5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sbls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`sbls_inform_type <doxid-structsbls__inform__type>`)
