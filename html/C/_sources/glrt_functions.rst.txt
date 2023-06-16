.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_glrt_control_type.rst
	struct_glrt_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`glrt_control_type<doxid-structglrt__control__type>`;
	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`;

	// global functions

	void :ref:`glrt_initialize<doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32>`(
		void** data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		int* status
	);

	void :ref:`glrt_read_specfile<doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725>`(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		const char specfile[]
	);

	void :ref:`glrt_import_control<doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1>`(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		void** data,
		int* status
	);

	void :ref:`glrt_solve_problem<doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>`(
		void** data,
		int* status,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` r[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` vector[]
	);

	void :ref:`glrt_information<doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d>`(void** data, struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform, int* status);

	void :ref:`glrt_terminate<doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812>`(
		void** data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform
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

.. index:: pair: function; glrt_initialize
.. _doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_initialize(
		void** data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
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

		- is a struct containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; glrt_read_specfile
.. _doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_read_specfile(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNGLRT.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/glrt.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; glrt_import_control
.. _doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_import_control(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		void** data,
		int* status
	)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; glrt_solve_problem
.. _doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_solve_problem(
		void** data,
		int* status,
		int n,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` power,
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` weight,
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` x[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` r[],
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` vector[]
	)

Solve the regularized-quadratic problem using reverse communication.



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
		  
		  * 1. on initial entry. Set r (below) to :math:`c` for this entry.
		  
		  * 6. the iteration is to be restarted with a larger weight but with all other data unchanged. Set r (below) to :math:`c` for this entry.
		  
		  Possible exit values are:
		  
		  * 0. the solution has been found
		  
		  * 2. the inverse of :math:`M` must be applied to vector with the result returned in vector and the function re-entered with all other data unchanged. This will only happen if control.unitm is false
		  
		  * 3. the product :math:`H` \* vector must be formed, with the result returned in vector and the function re-entered with all other data unchanged
		  
		  * 4. The iteration must be restarted. Reset r (below) to :math:`c` and re-enter with all other data unchanged.
		  
		  * -1. an array allocation has failed
		  
		  * -2. an array deallocation has failed
		  
		  * -3. n and/or radius is not positive
		  
		  * -7. the problem is unbounded from below. This can only happen if power = 2, and in this case the objective is unbounded along the arc x + t vector as t goes to infinity
		  
		  * -15. the matrix :math:`M` appears to be indefinite
		  
		  * -18. the iteration limit has been exceeded

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- power

		- is a scalar of type double, that holds the egularization power, :math:`p \geq 2`

	*
		- weight

		- is a scalar of type double, that holds the positive regularization weight, :math:`\sigma`

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- r

		- is a one-dimensional array of size n and type double, that that must be set to :math:`c` on entry (status = 1) and re-entry (status = 4, 5). On exit, r contains the resiual :math:`H x + c`.

	*
		- vector

		- is a one-dimensional array of size n and type double, that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; glrt_information
.. _doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_information(void** data, struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform, int* status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`glrt_inform_type <doxid-structglrt__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; glrt_terminate
.. _doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_terminate(
		void** data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`glrt_inform_type <doxid-structglrt__inform__type>`)

