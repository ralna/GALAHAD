.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_bnls_control_type.rst
	struct_bnls_inform_type.rst
	struct_bnls_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`bnls_control_type<doxid-structbnls__control__type>`;
	struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`;
	struct :ref:`bnls_time_type<doxid-structbnls__time__type>`;

	// function calls

	void :ref:`bnls_initialize<doxid-galahad__bnls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f>`(
		void **data,
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform
	);

	void :ref:`bnls_read_specfile<doxid-galahad__bnls_8h_1adf9db7eff2fce137ae2abd2e013c47b3>`(struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control, const char specfile[]);

	void :ref:`bnls_import<doxid-galahad__bnls_8h_1a3f0eb83fd31ee4108156f2e84176389d>`(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` cohort[],
	);

	void :ref:`bnls_import_withot_jac<doxid-galahad__bnls_8h_1a3f0eb83fd31ee4108156f2e84176390d>`(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` cohort[],
	);

	void :ref:`bnls_reset_control<doxid-galahad__bnls_8h_1a07f0857c9923ad0f92d51ed00833afda>`(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`bnls_solve_with_jac<doxid-galahad__bnls_8h_1ae923c2e6afabb3563fe0998d45b715c4>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` j_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_jr,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]

	);

	void :ref:`bnls_solve_with_jacprod<doxid-galahad__bnls_8h_1a692ecbfaa428584e60aa4c33d7278a64>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_jr_prod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, bool, const void*) eval_scol,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`ipc_<doxid-galahad__ipc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, bool, const void*) eval_sprod,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	);

	void :ref:`bnls_solve_reverse_with_jac<doxid-galahad__bnls_8h_1a9ad89605640c53c33ddd5894b5e3edd1>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` jr_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` Jr_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	);

	void :ref:`bnls_solve_reverse_with_jacprod<doxid-galahad__bnls_8h_1a6dddd928c19adec0abf76bdb2d75da17>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__rpc_8h_>` iv[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *lvl,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *lvu,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *index,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` p[],
		const :ref:`ipc_<doxid-galahad__rpc_8h_>` ip[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` lp,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	);

	void :ref:`bnls_information<doxid-galahad__bnls_8h_1a765da96b0a1f3d07dab53cc3400c22d8>`(void **data, struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`bnls_terminate<doxid-galahad__bnls_8h_1a7babe9112dfad1eb7b57b70135704ab0>`(
		void **data,
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform
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

function and structure names
----------------------------

The function and structure names described below are appropriate for the
default real working precision (``double``) and integer word length 
(``int32_t``). To use the functions and structures with different precisions
and integer word lengths, an additional suffix must be added to their names 
(and the arguments set accordingly). The appropriate suffices are:

``_s`` for single precision (``float``) reals and 
standard 32-bit (``int32_t``) integers;

``_q`` for quadruple precision (``__real128``) reals (if supported) and 
standard 32-bit (``int32_t``) integers;

``_64`` for standard precision (``double``) reals and 
64-bit (``int64_t``) integers;

``_s_64`` for single precision (``float``) reals and 
64-bit (``int64_t``) integers; and

``_q_64`` for quadruple precision (``__real128``) reals (if supported) and 
64-bit (``int64_t``) integers.

Thus a call to ``bnls_initialize`` below will instead be

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_initialize_s_64(void **data, struct :ref:`bnls_control_type_s_64<doxid-structbnls__control__type>`* control, 
                             int64_t *status)

if single precision (``float``) reals and 64-bit (``int64_t``) integers are 
required. Thus it is possible to call functions for this package 
with more that one precision and/or integer word length at same time. An 
:ref:`example<expo-multi-calls>` is provided for the package ``expo``,
and the obvious modifications apply equally here.

function calls
--------------

.. index:: pair: function; bnls_initialize
.. _doxid-galahad__bnls_8h_1aa344bb15b74ab3b3ee6afb2de072b19f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_initialize(
		void **data,
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`bnls_inform_type <doxid-structbnls__inform__type>`)

.. index:: pair: function; bnls_read_specfile
.. _doxid-galahad__bnls_8h_1adf9db7eff2fce137ae2abd2e013c47b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_read_specfile(struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/bnls/BNLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/bnls.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; bnls_import
.. _doxid-galahad__bnls_8h_1a3f0eb83fd31ee4108156f2e84176389d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_import(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		const char J_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` J_ptr[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` cohort[]
	)

Import problem data into internal storage prior to solution.


.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

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
                    The restrictions n > 0, m_r > 0 or requirement that
                    Jr_type contains its relevant string 'dense', 
                    'dense_by_rows', 'dense_by_columns', 'coordinate', 
                    'sparse_by_rows', or 'sparse_by_columns' has been
                    violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- J_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme<details-bnls_storage__unsym>` used for the Jacobian, $J_r$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- J_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in $J_r$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- J_row

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of $J_r$ in the sparse co-ordinate and sparse column-wise storage schemes. It need not be set for any of the remaining schemes, and in this case can be NULL.

	*
		- J_col

		- is a one-dimensional array of size J_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of $J_r$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set for any of the remaining schemes, and in this case can be NULL.

	*
		- J_ptr_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the length of the pointer array if sparse row or column storage scheme is used for $J_r$. For the sparse row scheme, Jr_ptr_ne should be at least   m_r+1, while for the sparse column scheme, it should be at least n+1, It should be set to 0 when the other schemes are used.

	*
		- J_ptr

		- is a one-dimensional array of size m+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of $J_r$, as well as the total number of entries, in the sparse row-wise storage scheme, or the starting position of each column of $J_r$, as well as the total number of entries, in the sparse column-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

	*
		- cohort

		- is a one-dimensional array of size m and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that specifies which cohort each variable is assigned to. If variable $x_j$ is associated with cohort $\cal C_i$, $0 \leq i \leq m_c-1$, cohort[j] should be set to i, while if $x_j$ is unconstrained cohort[j] = -1 should be assigned. At least one value cohort[j] for $j = 0,\ldots\,n-1$ is expected to take the value $i$ for every $0 \leq i \leq m_c-1$, that is no empty cohorts are allowed. If all the variables lie in a single simplex, cohort can be set to NULL.

.. index:: pair: function; bnls_import_without_jac
.. _doxid-galahad__bnls_8h_1a3f0eb83fd31ee4108156f2e84176390d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_import_without_jac(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` cohort[]
	)

Import problem data, excluding the structure of $J_r(x)$, into internal storage prior to solution.


.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

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
                    The restrictions n > 0, m_r > 0  has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- cohort

		- is a one-dimensional array of size m and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that specifies which cohort each variable is assigned to. If variable $x_j$ is associated with cohort $\cal C_i$, $0 \leq i \leq m_c-1$, cohort[j] should be set to i, while if $x_j$ is unconstrained cohort[j] = -1 should be assigned. At least one value cohort[j] for $j = 0,\ldots\,n-1$ is expected to take the value $i$ for every $0 \leq i \leq m_c-1$, that is no empty cohorts are allowed. If all the variables lie in a single simplex, cohort can be set to NULL.

.. index:: pair: function; bnls_reset_control
.. _doxid-galahad__bnls_8h_1a07f0857c9923ad0f92d51ed00833afda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_reset_control(
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; bnls_solve_with_jac
.. _doxid-galahad__bnls_8h_1ae923c2e6afabb3563fe0998d45b715c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_solve_with_jac(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` jr_ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_jr,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	)

Solve the simplex-constrained nonlinear least-squares problem when the Jacobian $J_r(x)$ is available by function calls.

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
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- y

		- is a one-dimensional array of size m_c and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $y$ of the Lagrange multipliers for the simplex constraints. The i-th component of y, i = 0, ... , m_r-1, contains $y_i$.

	*
		- z

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $z$ of the dual variables. The j-th component of z, j = 0, ... , n-1, contains $z_j$.

	*
		- r

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $r(x)$. The i-th component of r, j = 0, ... , n-1, contains $r_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat[j] is negative, the variable $x_j$ most likely lies at its zero lower bound,  while if it is zero, $x_j$ is free of its bound (or unconstrained).

	*
		- eval_r

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_r( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m_r, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], 
                         :ref:`rpc_<doxid-galahad__rpc_8h_>` r[], const void *userdata )
		  
		  The components of the residual function $r(x)$ evaluated at x= $x$ must be assigned to r, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_r`` via the structure ``userdata``.

	*
		- j_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the Jacobian matrix $J_r$.

	*
		- eval_jr

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_jr( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m, :ref:`ipc_<doxid-galahad__ipc_8h_>` jr_ne, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], :ref:`rpc_<doxid-galahad__rpc_8h_>` jr[],
		   	             const void *userdata )
		  
		  The components of the Jacobian $J_r = \nabla_x r(x$) of the residuals must be assigned to jr in the same order as presented to bnls_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jr`` via the structure ``userdata``.

	*
		- w

		- is a one-dimensional array of size m_r and type :ref:`rpc_<doxid-galahad__rpc_8h_>` that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

.. index:: pair: function; bnls_solve_with_jacprod
.. _doxid-galahad__bnls_8h_1a692ecbfaa428584e60aa4c33d7278a64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_solve_with_jacprod(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const void*) eval_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], bool, const void*) eval_jr_prod,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, bool, const void*) eval_scol,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`ipc_<doxid-galahad__ipc_8h_>`, :ref:`ipc_<doxid-galahad__ipc_8h_>`, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const bool, const :ref:`rpc_<doxid-galahad__rpc_8h_>`[], :ref:`rpc_<doxid-galahad__rpc_8h_>`[], const :ref:`ipc_<doxid-galahad__ipc_8h_>`[], :ref:`ipc_<doxid-galahad__ipc_8h_>`, bool, const void*) eval_sprod,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	)

Solve the simplex-constrained nonlinear least-squares problem when the products of the Jacobian $J_r(x)$ and its transpose are available by function calls.

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
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $r(x)$. The i-th component of r, j = 0, ... , n-1, contains $r_j(x)$.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat[j] is negative, the variable $x_j$ most likely lies at its zero lower bound,  while if it is zero, $x_j$ is free of its bound (or unconstrained).

	*
		- eval_r

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_r( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m_r, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], 
                         :ref:`rpc_<doxid-galahad__rpc_8h_>` r[], const void *userdata )
		  
		  The components of the residual function $r(x)$ evaluated at x= $x$ must be assigned to r, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_r`` via the structure ``userdata``.

	*
		- eval_jr_prod

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_jr_prod( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m_r, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], bool transpose,
		  	                   const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[],  :ref:`rpc_<doxid-galahad__rpc_8h_>` p[], bool got_jr,
		  	                   const void *userdata )
		   
		  The product $p = J_r(x) v$ (if tranpose is false) or $p = J_r^T(x) v$ (if tranpose is true) bewteen the product of the Jacobian $J_r(x) = \nabla_{x}c_(x)$ or its tranpose with the vector v= $v$ and the vector $ $u$ must be returned in p, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jr_prod`` via the structure ``userdata``.


	*
		- eval_jr_scol

		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp

		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_jr_scol( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m_r, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], 
		  	                   :ref:`ipc_<doxid-galahad__ipc_8h_>` index, :ref:`rpc_<doxid-galahad__rpc_8h_>` val[], 
		  	                   :ref:`ipc_<doxid-galahad__ipc_8h_>` row[], :ref:`ipc_<doxid-galahad__ipc_8h_>` nz, bool got_jr,
		  	                   const void *userdata )
 
		  The nonzeros and corresponding row entries of the index-th colum of $J_r(x)$ must be returned in val and row, respectively, together with the number of entries, nz, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jr_scol`` via the structure ``userdata``.

	*
		- eval_jr_sprod
		- 
		  is a user-supplied function that must have the following signature:
		  
		  .. ref-code-block:: cpp

		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_jr_sprod( :ref:`ipc_<doxid-galahad__ipc_8h_>` n, :ref:`ipc_<doxid-galahad__ipc_8h_>` m_r, const :ref:`rpc_<doxid-galahad__rpc_8h_>` x[], 
		  	                    const bool transpose, const :ref:`rpc_<doxid-galahad__rpc_8h_>` v[],  
		  	                    :ref:`rpc_<doxid-galahad__rpc_8h_>` p[], const :ref:`ipc_<doxid-galahad__ipc_8h_>` free[], :ref:`ipc_<doxid-galahad__ipc_8h_>` n_free, 
		  	                    bool got_jr, const void *userdata )                                   

		  The product $J_r(x) v$ (if tranpose is false) or $J_r^T(x) v$  (if tranpose is true) bewteen the Jacobian $J_r(x) = \nabla_{x}r(x)$ or its tranpose with the vector v=$v$ must be returned in p, and the function return value set to 0. If transpose is false, only the components free[0 : n_free-1] of $v$ will be nonzero, while if transpose is true, only the components free[0 : n_free-1] of p should be set. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_jr_sprod`` via the structure ``userdata``.

	*
		- w

		- is a one-dimensional array of size m_r and type :ref:`rpc_<doxid-galahad__rpc_8h_>` that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.

.. index:: pair: function; bnls_solve_reverse_with_jac
.. _doxid-galahad__bnls_8h_1a9ad89605640c53c33ddd5894b5e3edd1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_solve_reverse_with_jac(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` jr_ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` Jr_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	)

Solve the simplex-constrained nonlinear least-squares problem when the Jacobian $J_r(x)$ may be computed by the calling program.

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
                    $r(x)$ at the point $x$ indicated in x
                    and then re-enter the function. The required value
                    should be set in r, and eval_status should be set
                    to 0. If the user is unable to evaluate $r(x)$
                    for instance, if the function is undefined at
                    $x$ the user need not set r, but should then
                    set eval_status to a non-zero value.
		  
		  * **3**
                    The user should compute the Jacobian of the vector
                    of residual functions, $J_r(x) = \nabla_x r(x)$, at
                    the point $x$ indicated in x and then re-enter the
                    function. The l-th component of the Jacobian stored
                    according to the scheme specified for the remainder
                    of $J_r$ in the earlier call to bnls_import
                    should be set in Jr_val[l], for l = 0, ..., Jr_ne-1
                    and eval_status should be set to 0. If the user is
                    unable to evaluate a component of $J_r$ for
                    instance, if a component of the matrix is undefined
                    at $x$ the user need not set Jr_val, but should
                    then set eval_status to a non-zero value.

	*
		- eval_status

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $r(x)$. The i-th component of r, j = 0, ... , n-1, contains $r_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat[j] is negative, the variable $x_j$ most likely lies at its zero lower bound,  while if it is zero, $x_j$ is free of its bound (or unconstrained).

	*
		- jr_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the Jacobian matrix $J_r$.

	*
		- Jr_val

		- is a one-dimensional array of size jr_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the Jacobian matrix $J_r$ in any of the available storage schemes. See status = 3, above, for more details.

	*
		- w

		- is a one-dimensional array of size m_r and type :ref:`rpc_<doxid-galahad__rpc_8h_>` that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.
.. index:: pair: function; bnls_solve_reverse_with_jacprod
.. _doxid-galahad__bnls_8h_1a6dddd928c19adec0abf76bdb2d75da17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_solve_reverse_with_jacprod(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_r,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m_c,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` y[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` z[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` g[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` x_stat[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__rpc_8h_>` iv[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *lvl,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *lvu,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *index,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` p[],
		const :ref:`ipc_<doxid-galahad__rpc_8h_>` ip[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` lp,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` w[]
	)

Solve the simplex-constrained nonlinear least-squares problem when the  products of the Jacobian $J_r(x)$ and its transpose with specified vectors may be computed by the calling program.

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
                    $r(x)$ at the point $x$ indicated in x
                    and then re-enter the function. The required value
                    should be set in r, and eval_status should be set
                    to 0. If the user is unable to evaluate $r(x)$
                    for instance, if the function is undefined at
                    $x$ the user need not set r, but should then
                    set eval_status to a non-zero value.
		  
		  * **4**
                    The user should compute the product $p = J_r(x) v$, at
                    the point $x$ indicated in x, between the product of the 
                    Jacobian $J_r(x) = \nabla_{x}c_(x)$ with the vector v= $v$, 
                    and then re-enter the function. The result should be
                    set in p, and eval_status should be set to 0.
                    If the user is unable to evaluate the product, for 
                    instance, if the Jacobian is undefined at $x$ the user
                    need not set p, but should then set eval_status to a
                    non-zero value.
		  
		  * **5**
                    The user should compute the product $p = J_r^T(x) v$,
                    at the point $x$ indicated in x, between the product
                    of the transpose of the Jacobian $J_r(x) = \nabla_{x}c_(x)$ 
                    with the vector v= $v$, and then re-enter the function. The
                    result should be set in p, and eval_status should be set to 0. 
                    If the user is unable to evaluate the product, for 
                    instance, if the Jacobian is undefined at $x$ the user
                    need not set p, but should then set eval_status to a
                    non-zero value.
		  
		  * **6**
                    The user should compute the $j$-th column of $J_r(x)$, 
                    with $j$ provided in index, at the point $x$ given in x.
                    The resulting `nonzeros` and their corresponding row 
                    indices of the $j$-th column of $J_r(x)$ must be placed in 
                    p[0:lp-1] and ip[0:lp-1] with lp set accordingly,
                    and eval_status should be set to 0. 
                    If the user is unable to evaluate the column, for 
                    instance, if the Jacobian is undefined at $x$ the user
                    need not set p, ip and nz but should then set eval_status 
                    to a  non-zero value.

		  * **7**
                    The user should compute the product $p = J_r(x) v$
                    involving the residual Jacobian $J_r(x)$ at the point $x$, 
                    given in x, and a given sparse vector $v$, whose nonzeros
                    are in positions iv[lvl:lvu] of v. The resulting $p$ should
                    be placed in p and eval_status should be set to 0. 
                    If the user is unable to evaluate the product, for 
                    instance, if the Jacobian is undefined at $x$ the user
                    need not set p, but should then set eval_status to a
                    non-zero value.

		  * **8**
                    The user should compute selected components of the product 
                    $p = J_r^T(x) v$ involving the transpose of the residual 
                    Jacobian $J_r(x)$ at the point $x$, given in x, and a 
                    given vector $v$. Only components iv[lvl:lvu] of $p$ 
                    should be computed, and recorded in p[iv[lvl:lvu]], 
                    and eval_status should be set to 0. 
                    If the user is unable to evaluate the product, for 
                    instance, if the Jacobian is undefined at $x$ the user
                    need not set p, but should then set eval_status to a
                    non-zero value.

		  * **9**
                    The user has the opportunity to replace the estimate $x$ in 
                    x by an improved value $x^+$ for which $f(x^+) \leq f(x)$; 
                    in that case r must also be reset to hold $r(x^+)$.

	*
		- eval_status

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- m_r

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of residuals.

	*
		- m_c

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of cohorts.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the residual $r(x)$. The i-th component of r, j = 0, ... , n-1, contains $r_j(x)$. See status = 2, above, for more details.

	*
		- g

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the gradient $g = \nabla_xf(x)$ of the objective function. The j-th component of g, j = 0, ... , n-1, contains $g_j$.

	*
		- x_stat

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the optimal status of the problem variables. If x_stat[j] is negative, the variable $x_j$ most likely lies at its zero lower bound,  while if it is zero, $x_j$ is free of its bound (or unconstrained).

	*
		- v

		- is a one-dimensional array of size max(n,m_r) and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 4, 5, 7 and 8 above for more details.

	*
		- iv

		- is a one-dimensional array of size max(n,m_r) and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 7 and 8 above for more details.

	*
		- lvl

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 7 and 8 above for more details.

	*
		- lvu

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 7 and 8 above for more details.

	*
		- index

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 6 above for more details.

	*
		- p

		- is a one-dimensional array of size max(n,m_r) and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that is used for reverse communication. See status = 4 to 8 above for more details.

	*
		- ip

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 6 above for more details.

	*
		- lp

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used for reverse communication. See status = 6 above for more details.

	*
		- w

		- is a one-dimensional array of size m_r and type :ref:`rpc_<doxid-galahad__rpc_8h_>` that holds the values $w$ of the weights on the residuals in the least-squares objective function. It need not be set if the weights are all ones, and in this case can be NULL.
.. index:: pair: function; bnls_information
.. _doxid-galahad__bnls_8h_1a765da96b0a1f3d07dab53cc3400c22d8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_information(void **data, struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`bnls_inform_type <doxid-structbnls__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; bnls_terminate
.. _doxid-galahad__bnls_8h_1a7babe9112dfad1eb7b57b70135704ab0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void bnls_terminate(
		void **data,
		struct :ref:`bnls_control_type<doxid-structbnls__control__type>`* control,
		struct :ref:`bnls_inform_type<doxid-structbnls__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`bnls_control_type <doxid-structbnls__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`bnls_inform_type <doxid-structbnls__inform__type>`)

