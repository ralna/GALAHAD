.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_trek_control_type.rst
	struct_trek_time_type.rst
	struct_trek_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block
	
	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`trek_control_type<doxid-structtrek__control__type>`;
	struct :ref:`trek_time_type<doxid-structtrek__time__type>`;
	struct :ref:`trek_inform_type<doxid-structtrek__inform__type>`;

	// global functions

	void :ref:`trek_initialize<doxid-galahad__trek_8h_1acb066d992c4ec394402bc7b7317e1163>`(void **data, struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`trek_read_specfile<doxid-galahad__trek_8h_1adc7c56e7be2f7cc9d32921582d379b13>`(struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control, const char specfile[]);

	void :ref:`trek_import<doxid-galahad__trek_8h_1a4becded30e9b95fe7028b7799292c0af>`(
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	);

	void :ref:`trek_import_s<doxid-galahad__trek_8h_1a427420b6025d522bb7b3c652e8c2be48>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char S_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_ptr[]
	);

	void :ref:`trek_reset_control<doxid-galahad__trek_8h_1aae677e64bacb35354f49326815b694c3>`(
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`trek_solve_problem<doxid-galahad__trek_8h_1aadb8a751c29efcef663bf9560a1f9a8e>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` S_val[]
	);

	void :ref:`trek_information<doxid-galahad__trek_8h_1a3dda24010e564e2d6536cc7ea518451e>`(void **data, struct :ref:`trek_inform_type<doxid-structtrek__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`trek_terminate<doxid-galahad__trek_8h_1ab5cf0077db0631814fdd03599a585376>`(
		void **data,
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		struct :ref:`trek_inform_type<doxid-structtrek__inform__type>`* inform
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

Thus a call to ``trek_initialize`` below will instead be

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_initialize_s_64(void **data, struct :ref:`trek_control_type_s_64<doxid-structtrek__control__type>`* control, 
                             int64_t *status)

if single precision (``float``) reals and 64-bit (``int64_t``) integers are 
required. Thus it is possible to call functions for this package 
with more that one precision and/or integer word length at same time. An 
:ref:`example<expo-multi-calls>` is provided for the package ``expo``,
and the obvious modifications apply equally here.

function calls
--------------

.. index:: pair: function; trek_initialize
.. _doxid-galahad__trek_8h_1acb066d992c4ec394402bc7b7317e1163:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_initialize(void **data, struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`trek_control_type <doxid-structtrek__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; trek_read_specfile
.. _doxid-galahad__trek_8h_1adc7c56e7be2f7cc9d32921582d379b13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_read_specfile(struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list 
of keywords with associated default values is provided in 
\$GALAHAD/src/trek/TREK.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/trek.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`trek_control_type <doxid-structtrek__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; trek_import
.. _doxid-galahad__trek_8h_1a4becded30e9b95fe7028b7799292c0af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_import(
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`trek_control_type <doxid-structtrek__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
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
                    The restrictions n > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', diagonal',
                    'scaled-identity', 'identity', 'zero' or 'none'  has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows (and columns) of H.

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme<details-trek_storage__sym>` used for the Hessian, $H$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal', 'scaled-identity', 'identity', 'zero' or 'none'; lower or upper case variants are allowed.

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- H_col

		- is a one-dimensional array of size H_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $H$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $H$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; trek_import_s
.. _doxid-galahad__trek_8h_1a427420b6025d522bb7b3c652e8c2be48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_import_s(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char S_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` S_ptr[]
	)

Import data for the scaling matrix $S$ into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
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
                    'scaled-identity' or 'identity' has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of rows (and columns) of M.

	*
		- S_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme<details-trek_storage__sym>` used for the scaling matrix, $S$. It should be one of 'coordinate', 'sparse_by_rows', 'dense', or 'diagonal'; lower or upper case variants are allowed.

	*
		- S_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- S_row

		- is a one-dimensional array of size S_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- S_col

		- is a one-dimensional array of size S_ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $S$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or identity storage schemes are used, and in this case can be NULL.

	*
		- S_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $S$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; trek_reset_control
.. _doxid-galahad__trek_8h_1aae677e64bacb35354f49326815b694c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_reset_control(
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`trek_control_type <doxid-structtrek__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * 0. The import was successful.

.. index:: pair: function; trek_solve_problem
.. _doxid-galahad__trek_8h_1aadb8a751c29efcef663bf9560a1f9a8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_solve_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` H_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` S_ne,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` S_val[]
	)

Solve the trust-region subproblem.

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
                    The restrictions n > 0 or radius > 0, or
                    requirement that a type contains its relevant string
                    'dense', 'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity' or 'none' has been violated.
		  
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
                    A resolve call has been made before an initial call
                    (see control.new_radius and control.new_values).

                  * **-38**
                    An error occurred in a call to an LAPACK subroutine.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- H_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size h_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $c$ of the objective function. The j-th component of c, j = 0, ... , n-1, contains $c_j$.

	*
		- radius

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the trust-region radius, $\Delta$, used. radius must be strictly positive

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- S_ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the scaling matrix $S$ if it not the identity matrix.

	*
		- S_val

		- is a one-dimensional array of size S_ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the scaling matrix $S$, if it is not the identity matrix, in any of the available storage schemes. If S_val is NULL, $S$ will be taken to be the identity matrix.

.. index:: pair: function; trek_information
.. _doxid-galahad__trek_8h_1a3dda24010e564e2d6536cc7ea518451e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_information(void **data, struct :ref:`trek_inform_type<doxid-structtrek__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`trek_inform_type <doxid-structtrek__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; trek_terminate
.. _doxid-galahad__trek_8h_1ab5cf0077db0631814fdd03599a585376:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void trek_terminate(
		void **data,
		struct :ref:`trek_control_type<doxid-structtrek__control__type>`* control,
		struct :ref:`trek_inform_type<doxid-structtrek__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`trek_control_type <doxid-structtrek__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`trek_inform_type <doxid-structtrek__inform__type>`)

