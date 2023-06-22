.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_dps_control_type.rst
	struct_dps_inform_type.rst
	struct_dps_time_type.rst

.. _details-global:

function calls
--------------

.. ---------------------------------------------------------------------------
.. _global:

Global Scope
============


.. toctree::
	:hidden:

	namespace_conf.rst
	struct_dps_control_type.rst
	struct_dps_inform_type.rst
	struct_dps_time_type.rst

Overview
~~~~~~~~



.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	-- modules

	:ref:`conf<doxid-namespaceconf>`

	-- table types

	:ref:`dps_control_type<doxid-structdps__control__type>`
	:ref:`dps_inform_type<doxid-structdps__inform__type>`
	:ref:`dps_time_type<doxid-structdps__time__type>`

	-- functions

	function :ref:`dps_initialize<doxid-galahad__dps_8h_1a29104b1214a3af5b4dc76dca722250b4>`(data, control, status)
	function :ref:`dps_read_specfile<doxid-galahad__dps_8h_1a2b7fed0d89483ec1c49b517be04acdcf>`(control, specfile)
	function :ref:`dps_import<doxid-galahad__dps_8h_1a7bc05b1c7fd874e96481d0521262bdee>`(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
	function :ref:`dps_reset_control<doxid-galahad__dps_8h_1a445d31a1c3e3aa63af85ceddd9769a5c>`(control, data, status)
	function :ref:`dps_solve_tr_problem<doxid-galahad__dps_8h_1a0ce2d73010a90e735fd98393d63cb1a5>`(data, status, n, ne, H_val, c, f, radius, x)

	function :ref:`dps_solve_rq_problem<doxid-galahad__dps_8h_1ae3baff5b8a4b59c37a6ada62dff67cc6>`(
		data,
		status,
		n,
		ne,
		H_val,
		c,
		f,
		power,
		weight,
		x
		)

	function :ref:`dps_resolve_tr_problem<doxid-galahad__dps_8h_1af244a0e386040d5da2d11c3bd9d1e34d>`(data, status, n, c, f, radius, x)
	function :ref:`dps_resolve_rq_problem<doxid-galahad__dps_8h_1a19e02a1d80eaedcb9e339f9963db352a>`(data, status, n, c, f, power, weight, x)
	function :ref:`dps_information<doxid-galahad__dps_8h_1a7617a692133347cb651f9a96244eb9f6>`(data, inform, status)
	function :ref:`dps_terminate<doxid-galahad__dps_8h_1a1e67ac91c520fc4ec65df30e4140f57e>`(data, control, inform)

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Functions
---------

.. index:: pair: function; dps_initialize
.. _doxid-galahad__dps_8h_1a29104b1214a3af5b4dc76dca722250b4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_initialize(data, control, status)

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
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; dps_read_specfile
.. _doxid-galahad__dps_8h_1a2b7fed0d89483ec1c49b517be04acdcf:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNDPS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/dps.pdf for a list of keywords that may be set.



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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)

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
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was succesful, and the package is ready for the solve phase
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- H_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'; lower or upper case variants are allowed

	*
		- ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type int, that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type int, that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL

.. index:: pair: function; dps_reset_control
.. _doxid-galahad__dps_8h_1a445d31a1c3e3aa63af85ceddd9769a5c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_reset_control(control, data, status)

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
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are:
		  
		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; dps_solve_tr_problem
.. _doxid-galahad__dps_8h_1a0ce2d73010a90e735fd98393d63cb1a5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_solve_tr_problem(data, status, n, ne, H_val, c, f, radius, x)

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
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The run was succesful
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.
		  
		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status
		  
		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.
		  
		  * -40. An error has occured when building the preconditioner.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`c` in the objective function. The j-th component of c, j = 0, ... , n-1, contains :math:`c_j`.

	*
		- f

		- is a scalar variable pointer of type double, that holds the value of the holds the constant term :math:`f` in the objective function.

	*
		- radius

		- is a scalar variable pointer of type double, that holds the value of the trust-region radius, :math:`\Delta > 0`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

.. index:: pair: function; dps_solve_rq_problem
.. _doxid-galahad__dps_8h_1ae3baff5b8a4b59c37a6ada62dff67cc6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_solve_rq_problem(
		data,
		status,
		n,
		ne,
		H_val,
		c,
		f,
		power,
		weight,
		x
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
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The run was succesful
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.
		  
		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status
		  
		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.
		  
		  * -40. An error has occured when building the preconditioner.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- ne

		- is a scalar variable of type int, that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size ne and type double, that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- c

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`c` in the objective function. The j-th component of c, j = 0, ... , n-1, contains :math:`c_j`.

	*
		- f

		- is a scalar variable pointer of type double, that holds the value of the holds the constant term :math:`f` in the objective function.

	*
		- weight

		- is a scalar variable pointer of type double, that holds the value of the regularization weight, :math:`\sigma > 0`.

	*
		- power

		- is a scalar variable pointer of type double, that holds the value of the regularization power, :math:`p \geq 2`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

.. index:: pair: function; dps_resolve_tr_problem
.. _doxid-galahad__dps_8h_1af244a0e386040d5da2d11c3bd9d1e34d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_resolve_tr_problem(data, status, n, c, f, radius, x)

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
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The run was succesful
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been violated.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- c

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`c` in the objective function. The j-th component of c, j = 0, ... , n-1, contains :math:`c_j`.

	*
		- f

		- is a scalar variable pointer of type double, that holds the value of the constant term :math:`f` in the objective function.

	*
		- radius

		- is a scalar variable pointer of type double, that holds the value of the trust-region radius, :math:`\Delta > 0`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

.. index:: pair: function; dps_resolve_rq_problem
.. _doxid-galahad__dps_8h_1a19e02a1d80eaedcb9e339f9963db352a:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_resolve_rq_problem(data, status, n, c, f, power, weight, x)

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
		  is a scalar variable of type int, that gives the exit status from the package.
		  
		  Possible values are:
		  
		  * 0. The run was succesful
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -16. The problem is so ill-conditioned that further progress is impossible.

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- c

		- is a one-dimensional array of size n and type double, that holds the linear term :math:`c` in the objective function. The j-th component of c, j = 0, ... , n-1, contains :math:`c_j`.

	*
		- f

		- is a scalar variable pointer of type double, that holds the value of the holds the constant term :math:`f` in the objective function.

	*
		- weight

		- is a scalar variable pointer of type double, that holds the value of the regularization weight, :math:`\sigma > 0`.

	*
		- power

		- is a scalar variable pointer of type double, that holds the value of the regularization power, :math:`p \geq 2`.

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

.. index:: pair: function; dps_information
.. _doxid-galahad__dps_8h_1a7617a692133347cb651f9a96244eb9f6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_information(data, inform, status)

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
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; dps_terminate
.. _doxid-galahad__dps_8h_1a1e67ac91c520fc4ec65df30e4140f57e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function dps_terminate(data, control, inform)

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

