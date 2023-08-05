.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_lsrt_control_type.rst
	struct_lsrt_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block
	
	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`;
	struct :ref:`lsrt_inform_type<doxid-structlsrt__inform__type>`;

	// global functions

	void :ref:`lsrt_initialize<doxid-galahad__lsrt_8h_1a9c5c14ddb34a5ea1becd133837da6544>`(
		void** data,
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		int* status
	);

	void :ref:`lsrt_read_specfile<doxid-galahad__lsrt_8h_1a07c4c60e1ab6ae67a4da710e2ed01ff0>`(
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		const char specfile[]
	);

	void :ref:`lsrt_import_control<doxid-galahad__lsrt_8h_1a09e39db33990f0c8a66480f54ba80f09>`(
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`lsrt_solve_problem<doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992>`(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[]
	);

	void :ref:`lsrt_information<doxid-galahad__lsrt_8h_1ad3895aabdb7f18f84d209b02287872be>`(void** data, struct :ref:`lsrt_inform_type<doxid-structlsrt__inform__type>`* inform, int* status);

	void :ref:`lsrt_terminate<doxid-galahad__lsrt_8h_1ac3a3d73e2686538802563c795a1afff4>`(
		void** data,
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		struct :ref:`lsrt_inform_type<doxid-structlsrt__inform__type>`* inform
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

.. index:: pair: function; lsrt_initialize
.. _doxid-galahad__lsrt_8h_1a9c5c14ddb34a5ea1becd133837da6544:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_initialize(
		void** data,
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
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

		- is a struct containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was successful.

.. index:: pair: function; lsrt_read_specfile
.. _doxid-galahad__lsrt_8h_1a07c4c60e1ab6ae67a4da710e2ed01ff0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_read_specfile(
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNLSRT.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/lsrt.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; lsrt_import_control
.. _doxid-galahad__lsrt_8h_1a09e39db33990f0c8a66480f54ba80f09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_import_control(
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		void** data,
		int* status
	)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; lsrt_solve_problem
.. _doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_solve_problem(
		void** data,
		int* status,
		int m,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` u[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` v[]
	)

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
		  
		  * -3. one or more of n, m, power or weight violates allowed bounds
		  
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
		- x

		- is a one-dimensional array of size n and type double, that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- u

		- is a one-dimensional array of size m and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; lsrt_information
.. _doxid-galahad__lsrt_8h_1ad3895aabdb7f18f84d209b02287872be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_information(void** data, struct :ref:`lsrt_inform_type<doxid-structlsrt__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded successfully

.. index:: pair: function; lsrt_terminate
.. _doxid-galahad__lsrt_8h_1ac3a3d73e2686538802563c795a1afff4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lsrt_terminate(
		void** data,
		struct :ref:`lsrt_control_type<doxid-structlsrt__control__type>`* control,
		struct :ref:`lsrt_inform_type<doxid-structlsrt__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)

