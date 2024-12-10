.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_nls_subproblem_control_type.rst
	struct_nls_control_type.rst
	struct_nls_subproblem_inform_type.rst
	struct_nls_inform_type.rst
	struct_nls_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`nls_subproblem_control_type<doxid-structnls__subproblem__control__type>`;
	struct :ref:`nls_control_type<doxid-structnls__control__type>`;
	struct :ref:`nls_subproblem_inform_type<doxid-structnls__subproblem__inform__type>`;
	struct :ref:`nls_inform_type<doxid-structnls__inform__type>`;
	struct :ref:`nls_time_type<doxid-structnls__time__type>`;

	// function calls

	void :ref:`nls_initialize<doxid-galahad__nls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f>`(
		void **data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
	);

	void :ref:`nls_read_specfile<doxid-galahad__nls_8h_1adf9db7eff2fce137ae2abd2e013c47b3>`(struct :ref:`nls_control_type<doxid-structnls__control__type>`* control, const char specfile[]);

	void :ref:`nls_import<doxid-galahad__nls_8h_1a3f0eb83fd31ee4108156f2e84176389d>`(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char P_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` P_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_ptr[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	);

	void :ref:`nls_reset_control<doxid-galahad__nls_8h_1a07f0857c9923ad0f92d51ed00833afda>`(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`nls_solve_with_mat<doxid-galahad__nls_8h_1ae923c2e6afabb3563fe0998d45b715c4>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_c,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` j_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_j,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_h,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprods
	);

	void :ref:`nls_solve_without_mat<doxid-galahad__nls_8h_1a692ecbfaa428584e60aa4c33d7278a64>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_c,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_jprod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprods
	);

	void :ref:`nls_solve_reverse_with_mat<doxid-galahad__nls_8h_1a9ad89605640c53c33ddd5894b5e3edd1>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` j_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` J_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` P_val[]
	);

	void :ref:`nls_solve_reverse_without_mat<doxid-galahad__nls_8h_1a6dddd928c19adec0abf76bdb2d75da17>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		bool* transpose,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` P_val[]
	);

	void :ref:`nls_information<doxid-galahad__nls_8h_1a765da96b0a1f3d07dab53cc3400c22d8>`(void **data, struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`nls_terminate<doxid-galahad__nls_8h_1a7babe9112dfad1eb7b57b70135704ab0>`(
		void **data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
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

.. index:: pair: function; nls_initialize
.. _doxid-galahad__nls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_initialize(
		void **data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

.. index:: pair: function; nls_read_specfile
.. _doxid-galahad__nls_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_read_specfile(struct :ref:`nls_control_type<doxid-structnls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/nls/NLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/nls.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; nls_import
.. _doxid-galahad__nls_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_import(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[],
		const char P_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` P_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` P_ptr[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- J_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the Jacobian, $J$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- J_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- J_row

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $J$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- J_col

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $J$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- J_ptr

		- is a one-dimensional array of size m+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $J$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to $H$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- P_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the residual-Hessians-vector product matrix, $P$. It should be one of 'coordinate', 'sparse_by_columns', 'dense_by_columns' or 'absent', the latter if access to $P$ is via matrix-vector products; lower or upper case variants are allowed.

	*
		- P_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $P$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- P_row

		- is a one-dimensional array of size P_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $P$ in either the sparse co-ordinate, or the sparse column-wise storage scheme. It need not be set when the dense storage scheme is used, and in this case can be NULL.

	*
		- P_col

		- is a one-dimensional array of size P_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $P$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- P_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $P$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- w

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

.. index:: pair: function; nls_reset_control
.. _doxid-galahad__nls_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_reset_control(
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; nls_solve_with_mat
.. _doxid-galahad__nls_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_solve_with_mat(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_c,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` j_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_j,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_h,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprods
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where $H = \nabla_{xx}f(x)$ is provided specifically, and all function/derivative information is available by function calls.



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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- eval_c

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_c( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` c[], const void *userdata )
		  
		  The componnts of the residual function $c(x)$ evaluated at x= $x$ must be assigned to c, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_c`` via the structure ``userdata``.

	*
		- j_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the Jacobian matrix $J$.

	*
		- eval_j

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_j( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` jne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` j[],
		  	            const void *userdata )
		  
		  The components of the Jacobian $J = \nabla_x c(x$) of the residuals must be assigned to j in the same order as presented to nls_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_j`` via the structure ``userdata``.

	*
		- h_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$ if it is used.

	*
		- eval_h

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_h( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` hne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], const :ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		  	            :ref:`rpc_<doxid-galahad__rpc_8h_>` h[], const void *userdata )
		  
		  The nonzeros of the matrix $H = \sum_{i=1}^m y_i \nabla_{xx}c_i(x)$ of the weighted residual Hessian evaluated at x= $x$ and y= $y$ must be assigned to h in the same order as presented to nls_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_h`` via the structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the residual-Hessians-vector product matrix $P$ if it is used.

	*
		- eval_hprods

		- 
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_hprods( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` pne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		  	                    const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[], :ref:`rpc_<doxid-galahad__rpc_8h_>` p[], bool got_h,
		  	                    const void *userdata ) );
		  
		  The entries of the matrix $P$, whose i-th column is the product $\nabla_{xx}c_i(x) v$ between $\nabla_{xx}c_i(x)$, the Hessian of the i-th component of the residual $c(x)$ at x= $x$, and v= $v$ must be returned in p and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_without_mat
.. _doxid-galahad__nls_8h_1a692ecbfaa428584e60aa4c33d7278a64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_solve_without_mat(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_c,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_jprod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_hprods
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to $H = \nabla_{xx}f(x)$ is provided by Hessian-vector products, and all function/derivative information is available by function calls.



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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- eval_c

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_c( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` c[], const void *userdata )
		  
		  The componnts of the residual function $c(x)$ evaluated at x= $x$ must be assigned to c, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_c`` via the structure ``userdata``.

	*
		- eval_jprod

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_jprod( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], bool transpose,
		  	                :ref:`rpc_<doxid-galahad__rpc_8h_>` u[], const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[], bool got_j,
		  	                const void *userdata )
		  
		  The sum $u + \nabla_{x}c_(x) v$ (if tranpose is false) or The sum $u + (\nabla_{x}c_(x))^T v$ (if tranpose is true) bewteen the product of the Jacobian $\nabla_{x}c_(x)$ or its tranpose with the vector v= $v$ and the vector $ $u$ must be returned in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jprod`` via the structure ``userdata``.

	*
		- eval_hprod

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_hprod( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], const :ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		  	                :ref:`rpc_<doxid-galahad__rpc_8h_>` u[], const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[], bool got_h,
		  	                const void *userdata )
		  
		  The sum $u + \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v$ of the product of the weighted residual Hessian $H = \sum_{i=1}^m y_i \nabla_{xx}c_i(x)$ evaluated at x= $x$ and y= $y$ with the vector v= $v$ and the vector $ $u$ must be returned in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. The Hessians have already been evaluated or used at x if got_h is true. Data may be passed into ``eval_hprod`` via the structure ``userdata``.

	*
		- p_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the residual-Hessians-vector product matrix $P$ if it is used.

	*
		- eval_hprods

		- 
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_hprods( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		  	                 const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[], :ref:`rpc_<doxid-galahad__rpc_8h_>` pval[], bool got_h,
		  	                 const void *userdata )
		  
		  The entries of the matrix $P$, whose i-th column is the product $\nabla_{xx}c_i(x) v$ between $\nabla_{xx}c_i(x)$, the Hessian of the i-th component of the residual $c(x)$ at x= $x$, and v= $v$ must be returned in pval and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_hprods`` via the structure ``userdata``.

.. index:: pair: function; nls_solve_reverse_with_mat
.. _doxid-galahad__nls_8h_1a9ad89605640c53c33ddd5894b5e3edd1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_solve_reverse_with_mat(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` j_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` J_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` h_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` P_val[]
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where $H = \nabla_{xx}f(x)$ is provided specifically, but function/derivative information is only available by returning to the calling procedure



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
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
                    $c(x)$ at the point $x$ indicated in x
                    and then re-enter the function. The required value
                    should be set in c, and eval_status should be set
                    to 0. If the user is unable to evaluate $c(x)$
                    for instance, if the function is undefined at
                    $x$ the user need not set c, but should then
                    set eval_status to a non-zero value.
		  
		  * **3**
                    The user should compute the Jacobian of the vector
                    of residual functions, $\nabla_x c(x)$, at the
                    point $x$ indicated in x and then re-enter the
                    function. The l-th component of the Jacobian stored
                    according to the scheme specified for the remainder
                    of $J$ in the earlier call to nls_import
                    should be set in J_val[l], for l = 0, ..., J_ne-1
                    and eval_status should be set to 0. If the user is
                    unable to evaluate a component of $J$ for
                    instance, if a component of the matrix is undefined
                    at $x$ the user need not set J_val, but should
                    then set eval_status to a non-zero value.

		  * **4**
                    The user should compute the matrix $H =
                    \sum_{i=1}^m v_i \nabla_{xx}c_i(x)$ of weighted
                    residual Hessian evaluated at x= $x$ and v=
                    $v$ and then re-enter the function. The l-th
                    component of the matrix stored according to the
                    scheme specified for the remainder of $H$ in
                    the earlier call to nls_import should be set in
                    H_val[l], for l = 0, ..., H_ne-1 and eval_status
                    should be set to 0. If the user is unable to
                    evaluate a component of $H$ for instance, if a
                    component of the matrix is undefined at $x$
                    the user need not set H_val, but should then set
                    eval_status to a non-zero value. **Note** that this
                    return will not happen if the Gauss-Newton model is
                    selected.
		  
		  * **7**
                    The user should compute the entries of the matrix
                    $P$, whose i-th column is the product
                    $\nabla_{xx}c_i(x) v$ between
                    $\nabla_{xx}c_i(x)$, the Hessian of the i-th
                    component of the residual $c(x)$ at x=
                    $x$, and v= $v$ and then re-enter the
                    function. The l-th component of the matrix stored
                    according to the scheme specified for the remainder
                    of $P$ in the earlier call to nls_import
                    should be set in P_val[l], for l = 0, ..., P_ne-1
                    and eval_status should be set to 0. If the user is
                    unable to evaluate a component of $P$ for
                    instance, if a component of the matrix is undefined
                    at $x$ the user need not set P_val, but should
                    then set eval_status to a non-zero value. **Note**
                    that this return will not happen if either the
                    Gauss-Newton or Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- j_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the Jacobian matrix $J$.

	*
		- J_val

		- is a one-dimensional array of size j_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the Jacobian matrix $J$ in any of the available storage schemes. See status = 3, above, for more details.

	*
		- y

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 4 above for more details.

	*
		- h_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes. See status = 4, above, for more details.

	*
		- v

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 7, above, for more details.

	*
		- p_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the residual-Hessians-vector product matrix, $P$.

	*
		- P_val

		- is a one-dimensional array of size p_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the residual-Hessians-vector product matrix, $P$. See status = 7, above, for more details.

.. index:: pair: function; nls_solve_reverse_without_mat
.. _doxid-galahad__nls_8h_1a6dddd928c19adec0abf76bdb2d75da17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_solve_reverse_without_mat(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		bool* transpose,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` p_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` P_val[]
	)

Find a local minimizer of a given function using a trust-region method.

This call is for the case where access to $H = \nabla_{xx}f(x)$ is provided by Hessian-vector products, but function/derivative information is only available by returning to the calling procedure.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
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
                    $c(x)$ at the point $x$ indicated in x
                    and then re-enter the function. The required value
                    should be set in c, and eval_status should be set
                    to 0. If the user is unable to evaluate $c(x)$
                    for instance, if the function is undefined at
                    $x$ the user need not set c, but should then
                    set eval_status to a non-zero value.
		  
		  * **5**
                    The user should compute the sum $u +
                    \nabla_{x}c_(x) v$ (if tranpose is false) or
                    $u + (\nabla_{x}c_(x))^T v$ (if tranpose is
                    true) between the product of the Jacobian
                    $\nabla_{x}c_(x)$ or its tranpose with the
                    vector v= $v$ and the vector u = $ $u$,
                    and then re-enter the function. The result should be
                    set in u, and eval_status should be set to 0. If the
                    user is unable to evaluate the sum for instance, if
                    the Jacobian is undefined at $x$ the user need
                    not set u, but should then set eval_status to a
                    non-zero value.
		  
		  * **6**
                    The user should compute the sum $u +
                    \sum_{i=1}^m y_i \nabla_{xx}c_i(x) v$ between the
                    product of the weighted residual Hessian $H =
                    \sum_{i=1}^m y_i \nabla_{xx}c_i(x)$ evaluated at x=
                    $x$ and y= $y$ with the vector v=
                    $v$ and the the vector u = $ $u$, and
                    then re-enter the function. The result should be set
                    in u, and eval_status should be set to 0. If the
                    user is unable to evaluate the sum for instance, if
                    the weifghted residual Hessian is undefined at
                    $x$ the user need not set u, but should then
                    set eval_status to a non-zero value.
		  
		  * **7**
                    The user should compute the entries of the matrix
                    $P$, whose i-th column is the product
                    $\nabla_{xx}c_i(x) v$ between
                    $\nabla_{xx}c_i(x)$, the Hessian of the i-th
                    component of the residual $c(x)$ at x=
                    $x$, and v= $v$ and then re-enter the
                    function. The l-th component of the matrix stored
                    according to the scheme specified for the remainder
                    of $P$ in the earlier call to nls_import
                    should be set in P_val[l], for l = 0, ..., P_ne-1
                    and eval_status should be set to 0. If the user is
                    unable to evaluate a component of $P$ for
                    instance, if a component of the matrix is undefined
                    at $x$ the user need not set P_val, but should
                    then set eval_status to a non-zero value. **Note**
                    that this return will not happen if either the
                    Gauss-Newton or Newton models is selected.

	*
		- eval_status

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- c

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $c(x)$. The i-th component of c, j = 0, ... , n-1, contains $c_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- transpose

		- is a scalar variable of type bool, that indicates whether the product with Jacobian or its transpose should be obtained when status=5.

	*
		- u

		- is a one-dimensional array of size max(n,m) and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 5,6 above for more details.

	*
		- v

		- is a one-dimensional array of size max(n,m) and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 5,6,7 above for more details.

	*
		- y

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 6 above for more details.

	*
		- p_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the residual-Hessians-vector product matrix, $P$.

	*
		- P_val

		- is a one-dimensional array of size P_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the residual-Hessians-vector product matrix, $P$. See status = 7, above, for more details.

.. index:: pair: function; nls_information
.. _doxid-galahad__nls_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_information(void **data, struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; nls_terminate
.. _doxid-galahad__nls_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nls_terminate(
		void **data,
		struct :ref:`nls_control_type<doxid-structnls__control__type>`* control,
		struct :ref:`nls_inform_type<doxid-structnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`nls_control_type <doxid-structnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`nls_inform_type <doxid-structnls__inform__type>`)

