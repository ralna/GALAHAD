.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_llst_control_type.rst
	struct_llst_time_type.rst
	struct_llst_history_type.rst
	struct_llst_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block
	
	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`llst_control_type<doxid-structllst__control__type>`;
	struct :ref:`llst_history_type<doxid-structllst__history__type>`;
	struct :ref:`llst_inform_type<doxid-structllst__inform__type>`;
	struct :ref:`llst_time_type<doxid-structllst__time__type>`;

	// global functions

	void :ref:`llst_initialize<doxid-galahad__llst_8h_1a9da7a4daba2ceaf875fbd24fe42fbe1f>`(
		void** data,
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		int* status
	);

	void :ref:`llst_read_specfile<doxid-galahad__llst_8h_1a9bcda9a7420b5de742370e1464d5b0c2>`(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		const char specfile[]
	);

	void :ref:`llst_import<doxid-galahad__llst_8h_1a4ffc854176462b1d6492b55317150236>`(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char A_type[],
		int A_ne,
		const int A_row[],
		const int A_col[],
		const int A_ptr[]
	);

	void :ref:`llst_import_scaling<doxid-galahad__llst_8h_1a42d56aec0cdf37373e5a50b13b4c374f>`(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status,
		int n,
		const char S_type[],
		int S_ne,
		const int S_row[],
		const int S_col[],
		const int S_ptr[]
	);

	void :ref:`llst_reset_control<doxid-galahad__llst_8h_1a920e8696eea77dab3348a663a1127b41>`(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`llst_solve_problem<doxid-galahad__llst_8h_1add75b5481c528cca64abbcdeb3a2af35>`(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` radius,
		int A_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		int S_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` S_val[]
	);

	void :ref:`llst_information<doxid-galahad__llst_8h_1a88854815d1c936131dcc762c64275d6f>`(void** data, struct :ref:`llst_inform_type<doxid-structllst__inform__type>`* inform, int* status);

	void :ref:`llst_terminate<doxid-galahad__llst_8h_1a3d7693551362082a30094e7dea5a2a66>`(
		void** data,
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		struct :ref:`llst_inform_type<doxid-structllst__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; real_sp_
.. _doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float real_sp_

``real_sp_`` is real single precision

.. index:: pair: typedef; real_wp_
.. _doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double real_wp_

``real_wp_`` is the real working precision used

function calls
--------------

.. index:: pair: function; llst_initialize
.. _doxid-galahad__llst_8h_1a9da7a4daba2ceaf875fbd24fe42fbe1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_initialize(
		void** data,
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		int* status
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

		- is a struct containing control information (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; llst_read_specfile
.. _doxid-galahad__llst_8h_1a9bcda9a7420b5de742370e1464d5b0c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_read_specfile(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/llst/LLST.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/llst.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; llst_import
.. _doxid-galahad__llst_8h_1a4ffc854176462b1d6492b55317150236:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_import(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status,
		int m,
		int n,
		const char A_type[],
		int A_ne,
		const int A_row[],
		const int A_col[],
		const int A_ptr[]
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
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
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals, i.e., the number of rows of $A$. m must be positive.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables, i.e., the number of columns of $A$. n must be positive.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, $A$ if any. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in $A$, if used, in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- A_row

		- is a one-dimensional array of size A_ne and type int, that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be NULL.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; llst_import_scaling
.. _doxid-galahad__llst_8h_1a42d56aec0cdf37373e5a50b13b4c374f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_import_scaling(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status,
		int n,
		const char S_type[],
		int S_ne,
		const int S_row[],
		const int S_col[],
		const int S_ptr[]
	)

Import the scaling matrix $S$ into internal storage prior to solution. Thus must have been preceeded by a call to llst_import.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
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
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows' or 'diagonal' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables, i.e., the number of rows and columns of $S$. n must be positive.

	*
		- S_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid->` used for the matrix $S$. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or 'diagonal'; lower or upper case variants are allowed.

	*
		- S_ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- S_row

		- is a one-dimensional array of size S_ne and type int, that holds the row indices of the lower triangular part of $S$ in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL.

	*
		- S_col

		- is a one-dimensional array of size S_ne and type int, that holds the column indices of the lower triangular part of $S$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense, diagonal or (scaled) identity storage schemes are used, and in this case can be NULL.

	*
		- S_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of $S$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL.

.. index:: pair: function; llst_reset_control
.. _doxid-galahad__llst_8h_1a920e8696eea77dab3348a663a1127b41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_reset_control(
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		void** data,
		int* status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; llst_solve_problem
.. _doxid-galahad__llst_8h_1add75b5481c528cca64abbcdeb3a2af35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_solve_problem(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` radius,
		int A_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		int S_ne,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` S_val[]
	)

Solve the trust-region problem.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the entry and exit status from the package.
		  
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
                    The restrictions n > 0 and m > 0 or requirement that
                    A_type or A_type contains its relevant string
                    'dense', 'coordinate', 'sparse_by_rows' or
                    'diagonal' has been violated.
		  
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
		  
		  * **-15**
                    The matrix $S$ does not appear to be strictly
                    diagonally dominant.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.
		  
		  * **-17**
                    The step is too small to make further impact.

	*
		- m

		- is a scalar variable of type int, that holds the number of residuals

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- radius

		- is a scalar of type double, that holds the trust-region radius, $\Delta$, used. radius must be strictly positive

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of entries in the observation matrix $A$.

	*
		- A_val

		- is a one-dimensional array of size A_ne and type double, that holds the values of the entries of the observation matrix $A$ in any of the available storage schemes.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the values $b$ of observations. The i-th component of b, i = 0, ... , m-1, contains $b_i$.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- S_ne

		- is a scalar variable of type int, that holds the number of entries in the scaling matrix $S$ if it not the identity matrix.

	*
		- S_val

		- is a one-dimensional array of size S_ne and type double, that holds the values of the entries of the scaling matrix $S$ in any of the available storage schemes. If S_val is NULL, $S$ will be taken to be the identity matrix.

.. index:: pair: function; llst_information
.. _doxid-galahad__llst_8h_1a88854815d1c936131dcc762c64275d6f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_information(void** data, struct :ref:`llst_inform_type<doxid-structllst__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`llst_inform_type <doxid-structllst__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; llst_terminate
.. _doxid-galahad__llst_8h_1a3d7693551362082a30094e7dea5a2a66:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void llst_terminate(
		void** data,
		struct :ref:`llst_control_type<doxid-structllst__control__type>`* control,
		struct :ref:`llst_inform_type<doxid-structllst__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`llst_control_type <doxid-structllst__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`llst_inform_type <doxid-structllst__inform__type>`)

