.. _global:

.. index:: pair: function; ugo_initialize
.. _doxid-galahad__ugo_8h_1a172105bd528410f7c7e2fd77899ebc78:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_initialize(data, control, status)

Set default control values and initialize private data


.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; ugo_read_specfile
.. _doxid-galahad__ugo_8h_1a6819d58a728f3bf97232ed719e72fb91:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNUGO.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/ugo.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; ugo_import
.. _doxid-galahad__ugo_8h_1a8bcbdf9ef1229535b77d9991eb543dcb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_import(control, data, status, x_l, x_u)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

	*
		- x_l

		- is a scalar variable of type T, that holds the value :math:`x^l` of the lower bound on the optimization variable :math:`x`.

	*
		- x_u

		- is a scalar variable of type T, that holds the value :math:`x^u` of the upper bound on the optimization variable :math:`x`.

.. index:: pair: function; ugo_reset_control
.. _doxid-galahad__ugo_8h_1a51fa6faacfb75c3dcad44befd2e6cb40:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_reset_control(control, data, status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; ugo_solve_direct
.. _doxid-galahad__ugo_8h_1aa5b2949ab17e25a0a0c24f38c0d61a1a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_solve_direct(data, userdata, status, x, f, g, h, eval_fgh)

Find an approximation to the global minimizer of a given univariate function with a Lipschitz gradient in an interval.

This version is for the case where all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs (see below).

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -7. The objective function appears to be unbounded from below

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -40. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

	*
		- x

		- is a scalar variable of type T, that holds the value of the approximate global minimizer :math:`x` after a successful (status = 0) call.

	*
		- f

		- is a scalar variable of type T, that holds the the value of the objective function :math:`f(x)` at the approximate global minimizer :math:`x` after a successful (status = 0) call.

	*
		- g

		- is a scalar variable of type T, that holds the the value of the gradient of the objective function :math:`f^{\prime}(x)` at the approximate global minimizer :math:`x` after a successful (status = 0) call.

	*
		- h

		- is a scalar variable of type T, that holds the the value of the second derivative of the objective function :math:`f^{\prime\prime}(x)` at the approximate global minimizer :math:`x` after a successful (status = 0) call.

	*
		- eval_fgh

		-
		  is a user-provided function that must have the following signature:

		  .. ref-code-block:: julia

		  	eval_fgh(x, f, g, h, userdata)

		  The value of the objective function :math:`f(x)` and its first derivative :math:`f^{\prime}(x)` evaluated at x= :math:`x` must be assigned to f and g respectively, and the function return value set to 0. In addition, if control.second_derivatives_available has been set to true, when calling ugo_import, the user must also assign the value of the second derivative :math:`f^{\prime\prime}(x)` in h; it need not be assigned otherwise. If the evaluation is impossible at x, return should be set to a nonzero value.

.. index:: pair: function; ugo_solve_reverse
.. _doxid-galahad__ugo_8h_1a0b8f123f8e67bb0cb8a27c5ce87c824c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_solve_reverse(data, status, eval_status, x, f, g, h)

Find an approximation to the global minimizer of a given univariate function with a Lipschitz gradient in an interval.

This version is for the case where function/derivative information is only available by returning to the calling procedure.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -7. The objective function appears to be unbounded from below

		  * -18. Too many iterations have been performed. This may happen if control.maxit is too small, but may also be symptomatic of a badly scaled problem.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -40. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.



		  * 3. The user should compute the objective function value :math:`f(x)` and its first derivative :math:`f^{\prime}(x)`, and then re-enter the function. The required values should be set in f and g respectively, and eval_status (below) should be set to 0. If the user is unable to evaluate :math:`f(x)` or :math:`f^{\prime}(x)` - for instance, if the function or its first derivative are undefined at x - the user need not set f or g, but should then set eval_status to a non-zero value. This value can only occur when control.second_derivatives_available = false.



		  * 4. The user should compute the objective function value :math:`f(x)` and its first two derivatives :math:`f^{\prime}(x)` and :math:`f^{\prime\prime}(x)` at x= :math:`x`, and then re-enter the function. The required values should be set in f, g and h respectively, and eval_status (below) should be set to 0. If the user is unable to evaluate :math:`f(x)`, :math:`f^{\prime}(x)` or :math:`f^{\prime\prime}(x)` - for instance, if the function or its derivatives are undefined at x - the user need not set f, g or h, but should then set eval_status to a non-zero value. This value can only occur when control.second_derivatives_available = true.

	*
		- eval_status

		- is a scalar variable of type Int32, that is used to indicate if objective function and its derivatives can be provided (see above).

	*
		- x

		- is a scalar variable of type T, that holds the next value of :math:`x` at which the user is required to evaluate the objective (and its derivatives) when status > 0, or the value of the approximate global minimizer when status = 0

	*
		- f

		- is a scalar variable of type T, that must be set by the user to hold the value of :math:`f(x)` if required by status > 0 (see above), and will return the value of the approximate global minimum when status = 0

	*
		- g

		- is a scalar variable of type T, that must be set by the user to hold the value of :math:`f^{\prime}(x)` if required by status > 0 (see above), and will return the value of the first derivative of :math:`f` at the approximate global minimizer when status = 0

	*
		- h

		- is a scalar variable of type T, that must be set by the user to hold the value of :math:`f^{\prime\prime}(x)` if required by status > 0 (see above), and will return the value of the second derivative of :math:`f` at the approximate global minimizer when status = 0

.. index:: pair: function; ugo_information
.. _doxid-galahad__ugo_8h_1a8e1db35daea3247b2cc9eb8607d0abee:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_information(data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`ugo_inform_type <doxid-structugo__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32, that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; ugo_terminate
.. _doxid-galahad__ugo_8h_1ad9485926c547bb783aea3ee1adb3b084:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        ugo_terminate(data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`ugo_inform_type <doxid-structugo__inform__type>`)
