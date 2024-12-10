.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_dps_control_type.rst
	struct_dps_time_type.rst
	struct_dps_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`dps_control_type<doxid-structdps__control__type>`;
	struct :ref:`dps_inform_type<doxid-structdps__inform__type>`;
	struct :ref:`dps_time_type<doxid-structdps__time__type>`;

	// global functions

	void :ref:`dps_initialize<doxid-galahad__dps_8h_1a29104b1214a3af5b4dc76dca722250b4>`(void **data, struct :ref:`dps_control_type<doxid-structdps__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`dps_read_specfile<doxid-galahad__dps_8h_1a2b7fed0d89483ec1c49b517be04acdcf>`(struct :ref:`dps_control_type<doxid-structdps__control__type>`* control, const char specfile[]);

	void :ref:`dps_import<doxid-galahad__dps_8h_1a7bc05b1c7fd874e96481d0521262bdee>`(
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` H_ptr[]
	);

	void :ref:`dps_reset_control<doxid-galahad__dps_8h_1a445d31a1c3e3aa63af85ceddd9769a5c>`(
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`dps_solve_tr_problem<doxid-galahad__dps_8h_1a0ce2d73010a90e735fd98393d63cb1a5>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	);

	void :ref:`dps_solve_rq_problem<doxid-galahad__dps_8h_1ae3baff5b8a4b59c37a6ada62dff67cc6>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	);

	void :ref:`dps_resolve_tr_problem<doxid-galahad__dps_8h_1af244a0e386040d5da2d11c3bd9d1e34d>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	);

	void :ref:`dps_resolve_rq_problem<doxid-galahad__dps_8h_1a19e02a1d80eaedcb9e339f9963db352a>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	);

	void :ref:`dps_information<doxid-galahad__dps_8h_1a7617a692133347cb651f9a96244eb9f6>`(void **data, struct :ref:`dps_inform_type<doxid-structdps__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`dps_terminate<doxid-galahad__dps_8h_1a1e67ac91c520fc4ec65df30e4140f57e>`(
		void **data,
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		struct :ref:`dps_inform_type<doxid-structdps__inform__type>`* inform
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

.. index:: pair: function; dps_initialize
.. _doxid-galahad__dps_8h_1a29104b1214a3af5b4dc76dca722250b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_initialize(void **data, struct :ref:`dps_control_type<doxid-structdps__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`dps_control_type <doxid-structdps__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; dps_read_specfile
.. _doxid-galahad__dps_8h_1a2b7fed0d89483ec1c49b517be04acdcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_read_specfile(struct :ref:`dps_control_type<doxid-structdps__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/dps/DPS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/dps.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`dps_control_type <doxid-structdps__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; dps_import
.. _doxid-galahad__dps_8h_1a7bc05b1c7fd874e96481d0521262bdee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_import(
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
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

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`dps_control_type <doxid-structdps__control__type>`)

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
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate'
                    or 'sparse_by_rows' has been violated.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL

.. index:: pair: function; dps_reset_control
.. _doxid-galahad__dps_8h_1a445d31a1c3e3aa63af85ceddd9769a5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_reset_control(
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`dps_control_type <doxid-structdps__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; dps_solve_tr_problem
.. _doxid-galahad__dps_8h_1a0ce2d73010a90e735fd98393d63cb1a5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_solve_tr_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	)

Find the global minimizer of the trust-region problem (1).



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
                    contains its relevant string 'dense', 'coordinate'
                    or 'sparse_by_rows' has been violated.
		  
		  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status
		  
		  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.
		  
		  * **-40**
                    An error has occured when building the
                    preconditioner.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $c$ in the objective function. The j-th component of c, j = 0, ... , n-1, contains $c_j$.

	*
		- f

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the holds the constant term $f$ in the objective function.

	*
		- radius

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the trust-region radius, $\Delta > 0$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

.. index:: pair: function; dps_solve_rq_problem
.. _doxid-galahad__dps_8h_1ae3baff5b8a4b59c37a6ada62dff67cc6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_solve_rq_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` H_val[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	)

Find the global minimizer of the regularized-quadartic problem (2).



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
                    contains its relevant string 'dense', 'coordinate'
                    or 'sparse_by_rows' has been violated.
		  
		  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status
		  
		  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.
		  
		  * **-40**
                    An error has occured when building the
                    preconditioner.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- H_val

		- is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $c$ in the objective function. The j-th component of c, j = 0, ... , n-1, contains $c_j$.

	*
		- f

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the holds the constant term $f$ in the objective function.

	*
		- weight

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the regularization weight, $\sigma > 0$.

	*
		- power

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the regularization power, $p \geq 2$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

.. index:: pair: function; dps_resolve_tr_problem
.. _doxid-galahad__dps_8h_1af244a0e386040d5da2d11c3bd9d1e34d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_resolve_tr_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	)

Find the global minimizer of the trust-region problem (1) if some non-matrix components have changed since a call to dps_solve_tr_problem.



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
                    contains its relevant string 'dense', 'coordinate'
                    or 'sparse_by_rows' has been violated.
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- c

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $c$ in the objective function. The j-th component of c, j = 0, ... , n-1, contains $c_j$.

	*
		- f

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the constant term $f$ in the objective function.

	*
		- radius

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the trust-region radius, $\Delta > 0$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

.. index:: pair: function; dps_resolve_rq_problem
.. _doxid-galahad__dps_8h_1a19e02a1d80eaedcb9e339f9963db352a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_resolve_rq_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` c[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[]
	)

Find the global minimizer of the regularized-quadartic problem (2) if some non-matrix components have changed since a call to dps_solve_rq_problem.



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
		  
		  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- c

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the linear term $c$ in the objective function. The j-th component of c, j = 0, ... , n-1, contains $c_j$.

	*
		- f

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the holds the constant term $f$ in the objective function.

	*
		- weight

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the regularization weight, $\sigma > 0$.

	*
		- power

		- is a scalar variable pointer of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the regularization power, $p \geq 2$.

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values $x$ of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

.. index:: pair: function; dps_information
.. _doxid-galahad__dps_8h_1a7617a692133347cb651f9a96244eb9f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_information(void **data, struct :ref:`dps_inform_type<doxid-structdps__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`dps_inform_type <doxid-structdps__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; dps_terminate
.. _doxid-galahad__dps_8h_1a1e67ac91c520fc4ec65df30e4140f57e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dps_terminate(
		void **data,
		struct :ref:`dps_control_type<doxid-structdps__control__type>`* control,
		struct :ref:`dps_inform_type<doxid-structdps__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`dps_control_type <doxid-structdps__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`dps_inform_type <doxid-structdps__inform__type>`)

