.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_l2rt_control_type.rst
	struct_l2rt_inform_type.rst

function calls
--------------

.. index:: pair: function; l2rt_initialize
.. _doxid-galahad__l2rt_8h_1a0103448a3db662f9c483f9f44a5112bc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_initialize(
		void** data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
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

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; l2rt_read_specfile
.. _doxid-galahad__l2rt_8h_1a1b63f8b501208629cceb662b03f35684:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_read_specfile(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNL2RT.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/l2rt.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; l2rt_import_control
.. _doxid-galahad__l2rt_8h_1adf880b26c8aea32493857f8576e58ae8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_import_control(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		void** data,
		int* status
	)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; l2rt_solve_problem
.. _doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_solve_problem(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` shift,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[]
	)

Solve the regularized-least-squares problem using reverse communication.



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

		  * -3. one or more of n, m, weight or shift violates allowed bounds

		  * -18. the iteration limit has been exceeded

		  * -25. status is negative on entry

	*
		- m

		- is a scalar variable of type int, that holds the number of equations (i.e., rows of :math:`A`), :math:`m > 0`

	*
		- n

		- is a scalar variable of type int, that holds the number of variables (i.e., columns of :math:`A`), :math:`n > 0`

	*
		- power

		- is a scalar of type double, that holds the regularization power, :math:`p \geq 2`

	*
		- weight

		- is a scalar of type double, that holds the regularization weight, :math:`\sigma > 0`

	*
		- shift

		- is a scalar of type double, that holds the shift, :math:`\mu`

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- u

		- is a one-dimensional array of size m and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; l2rt_information
.. _doxid-galahad__l2rt_8h_1a4fa18245556cf87b255b2b9ac5748ca9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_information(void** data, struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`l2rt_inform_type <doxid-structl2rt__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; l2rt_terminate
.. _doxid-galahad__l2rt_8h_1aa9b62de33c3d6c129cca1e90a3d548b7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void l2rt_terminate(
		void** data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`l2rt_inform_type <doxid-structl2rt__inform__type>`)
