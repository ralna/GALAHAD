.. _global:

callable functions
------------------

.. index:: pair: function; lsrt_initialize
.. _doxid-galahad__lsrt_8h_1a9c5c14ddb34a5ea1becd133837da6544:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_initialize(data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; lsrt_read_specfile
.. _doxid-galahad__lsrt_8h_1a07c4c60e1ab6ae67a4da710e2ed01ff0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNLSRT.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/lsrt.pdf for a list of keywords that may be set.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; lsrt_import_control
.. _doxid-galahad__lsrt_8h_1a09e39db33990f0c8a66480f54ba80f09:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_import_control(control, data, status)

Import control parameters prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; lsrt_solve_problem
.. _doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_solve_problem(data, status, m, n, power, weight, x, u, v)

Solve the regularized least-squuares problem using reverse communication.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  This must be set to

		  * 1. on initial entry. Set u (below) to :math:`b` for this entry.

		  Possible exit values are:

		  * 0. the solution has been found

		  * 2. The user must perform the operation

		    .. math::

		    	u := u + A v,

		    \n
		                   u := u + A v,
		    \n and recall the function. The vectors :math:`u` and :math:`v` are available in the arrays u and v (below) respectively, and the result :math:`u` must overwrite the content of u. No argument except u should be altered before recalling the function

		  * 3. The user must perform the operation

		    .. math::

		    	v := v + A^T u,

		    \n
		                   v := v + A^T u,
		    \n and recall the function. The vectors :math:`u` and :math:`v` are available in the arrays u and v (below) respectively, and the result :math:`v` must overwrite the content of v. No argument except v should be altered before recalling the function

		  * 4. The user must reset u (below) to :math:`b` are recall the function. No argument except u should be altered before recalling the function

		  * -1. an array allocation has failed

		  * -2. an array deallocation has failed

		  * -3. one or more of n, m, power or weight violates allowed bounds

		  * -18. the iteration limit has been exceeded

		  * -25. status is negative on entry

	*
		- m

		- is a scalar variable of type Int32 that holds the number of equations (i.e., rows of :math:`A`), :math:`m > 0`

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables (i.e., columns of :math:`A`), :math:`n > 0`

	*
		- power

		- is a scalar of type T that holds the regularization power, :math:`p \geq 2`

	*
		- weight

		- is a scalar of type T that holds the regularization weight, :math:`\sigma > 0`

	*
		- x

		- is a one-dimensional array of size n and type T that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- u

		- is a one-dimensional array of size m and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; lsrt_information
.. _doxid-galahad__lsrt_8h_1ad3895aabdb7f18f84d209b02287872be:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_information(data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; lsrt_terminate
.. _doxid-galahad__lsrt_8h_1ac3a3d73e2686538802563c795a1afff4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_terminate(data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)
