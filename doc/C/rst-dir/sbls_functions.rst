.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sbls_control_type.rst
	struct_sbls_time_type.rst
	struct_sbls_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>`;
	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>`;
	struct :ref:`sbls_time_type<doxid-structsbls__time__type>`;

	// global functions

	void :ref:`sbls_initialize<doxid-galahad__sbls_8h_1a30b1a9463e4abd5cfa0150ffb30569a9>`(
		void **data,
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`sbls_read_specfile<doxid-galahad__sbls_8h_1abde2e76567a4c8721fe9c2386106e972>`(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		const char specfile[]
	);

	void :ref:`sbls_import<doxid-galahad__sbls_8h_1ab7cbabccf52f8be7ae417e089eba4b82>`(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[],
		const char C_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` C_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_ptr[]
	);

	void :ref:`sbls_reset_control<doxid-galahad__sbls_8h_1afdfe80ab659c2936d23802b6a6103eb8>`(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`sbls_factorize_matrix<doxid-galahad__sbls_8h_1a77799da1282c3567b56ae8db42b75f65>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` C_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` D[]
	);

	void :ref:`sbls_solve_system<doxid-galahad__sbls_8h_1a2c3ae7b15fc1c43771d395540c37b9fa>`(void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status, :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`rpc_<doxid-galahad__rpc_8h_>` sol[]);
	void :ref:`sbls_information<doxid-galahad__sbls_8h_1a9f93f5c87ae0088ceb72c4f7e73c9418>`(void **data, struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`sbls_terminate<doxid-galahad__sbls_8h_1a73d7d29d113a62c48cc176146539bca5>`(
		void **data,
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; spc_
.. _doxid-galahad__spc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float spc_

``spc_`` is real single precision

.. index:: pair: typedef; rpc_
.. _doxid-galahad__rpc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double rpc_

``rpc_`` is the real working precision used, but may be changed to ``float`` by
defining the  preprocessor variable ``REAL_32`` or (if supported) to
``__real128`` using the variable ``REAL_128``.

.. index:: pair: typedef; ipc_
.. _doxid-galahad__ipc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int ipc_

``ipc_`` is the default integer word length used, but may be changed to 
``int64_t`` by defining the  preprocessor variable ``INTEGER_64``.

function calls
--------------

.. index:: pair: function; sbls_initialize
.. _doxid-galahad__sbls_8h_1a30b1a9463e4abd5cfa0150ffb30569a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_initialize(
		void **data,
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; sbls_read_specfile
.. _doxid-galahad__sbls_8h_1abde2e76567a4c8721fe9c2386106e972:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_read_specfile(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list 
of keywords with associated default values is provided in 
\$GALAHAD/src/sbls/SBLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/sbls.pdf for a list of how these keywords
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; sbls_import
.. _doxid-galahad__sbls_8h_1ab7cbabccf52f8be7ae417e089eba4b82:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_import(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char A_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` A_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[],
		const char C_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` C_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` C_ptr[]
	)

Import structural matrix data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows in the symmetric matrix $H$.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows in the symmetric matrix $C$.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the matrix $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if $H=0$; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- C_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the matrix $C$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none', the latter pair if $C=0$; lower or upper case variants are allowed.

	*
		- C_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $C$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- C_row

		- is a one-dimensional array of size C_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $C$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- C_col

		- is a one-dimensional array of size C_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $C$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- C_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $C$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; sbls_reset_control
.. _doxid-galahad__sbls_8h_1afdfe80ab659c2936d23802b6a6103eb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_reset_control(
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **0**
                    The import was successful.

.. index:: pair: function; sbls_factorize_matrix
.. _doxid-galahad__sbls_8h_1a77799da1282c3567b56ae8db42b75f65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_factorize_matrix(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` a_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` A_val[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` c_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` C_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` D[]
	)

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

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
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
                    The computed preconditioner $K_G$ is singular
                    and is thus unsuitable.
		  
		  * **-20**
                    The computed preconditioner $K_G$ has the
                    wrong inertia and is thus unsuitable.
		  
		  * **-24**
                    An error was reported by the GALAHAD package
                    SORT_reorder_by_rows. The return status from
                    SORT_reorder_by_rows is given in
                    inform.sort_status. See the documentation for the
                    GALAHAD package SORT for further details.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows in the symmetric matrix $H$.

	*
		- h_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the symmetric matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the symmetric matrix $H$ in any of the available storage schemes

	*
		- a_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the unsymmetric matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the unsymmetric matrix $A$ in any of the available storage schemes.

	*
		- c_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the symmetric matrix $C$.

	*
		- C_val

		- is a one-dimensional array of size c_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the symmetric matrix $C$ in any of the available storage schemes

	*
		- D

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the diagonal matrix $D$ that is required if the user has specified control.preconditioner = 5. It need not be set otherwise.

.. index:: pair: function; sbls_solve_system
.. _doxid-galahad__sbls_8h_1a2c3ae7b15fc1c43771d395540c37b9fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_solve_system(void **data, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status, :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`rpc_<doxid-galahad__rpc_8h_>` sol[])

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

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package.
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the vector $a$.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the vector $b$.

	*
		- sol

		- is a one-dimensional array of size n + m and type double. on entry, its first n entries must hold the vector $a$, and the following entries must hold the vector $b$. On a successful exit, its first n entries contain the solution components $x$, and the following entries contain the components $y$.

.. index:: pair: function; sbls_information
.. _doxid-galahad__sbls_8h_1a9f93f5c87ae0088ceb72c4f7e73c9418:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_information(void **data, struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`sbls_inform_type <doxid-structsbls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sbls_terminate
.. _doxid-galahad__sbls_8h_1a73d7d29d113a62c48cc176146539bca5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sbls_terminate(
		void **data,
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>`* control,
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>`* inform
	)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`sbls_control_type <doxid-structsbls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`sbls_inform_type <doxid-structsbls__inform__type>`)

