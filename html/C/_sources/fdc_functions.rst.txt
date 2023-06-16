.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_fdc_control_type.rst
	struct_fdc_time_type.rst
	struct_fdc_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`real_sp_<doxid-galahad__precision_8h_1a3455cab03087949fd428a31cf302f98b>`;
	typedef double :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>`;

	// structs

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>`;
	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>`;
	struct :ref:`fdc_time_type<doxid-structfdc__time__type>`;

	// global functions

	void :ref:`fdc_initialize<doxid-galahad__fdc_8h_1a09ed47873fc4b54eac5b10958939459b>`(void** data, struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control, int* status);
	void :ref:`fdc_read_specfile<doxid-galahad__fdc_8h_1aa5e20e6a3ed015cdd927c1bfc7f00a2a>`(struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control, const char specfile[]);

	void :ref:`fdc_find_dependent_rows<doxid-galahad__fdc_8h_1a37ea723b9a1b8799e7971344858d020a>`(
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control,
		void** data,
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>`* inform,
		int* status,
		int m,
		int n,
		int A_ne,
		const int A_col[],
		const int A_ptr[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		int* n_depen,
		int depen[]
	);

	void :ref:`fdc_terminate<doxid-galahad__fdc_8h_1a9c0167379258891dee32b35e0529b9f9>`(
		void** data,
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control,
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>`* inform
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

.. index:: pair: function; fdc_initialize
.. _doxid-galahad__fdc_8h_1a09ed47873fc4b54eac5b10958939459b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fdc_initialize(void** data, struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; fdc_read_specfile
.. _doxid-galahad__fdc_8h_1aa5e20e6a3ed015cdd927c1bfc7f00a2a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fdc_read_specfile(struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNEQP.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/eqp.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; fdc_find_dependent_rows
.. _doxid-galahad__fdc_8h_1a37ea723b9a1b8799e7971344858d020a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fdc_find_dependent_rows(
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control,
		void** data,
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>`* inform,
		int* status,
		int m,
		int n,
		int A_ne,
		const int A_col[],
		const int A_ptr[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` A_val[],
		const :ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` b[],
		int* n_depen,
		int depen[]
	)

Find dependent rows and, if any, check if :math:`A x = b` is consistent



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`fdc_inform_type <doxid-structfdc__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the entry and exit status from the package.
		  
		  Possible exit are:
		  
		  * 0. The run was succesful.
		  
		  
		  
		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.
		  
		  * -3. The restrictions n > 0 and m > 0 or requirement that a type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal', 'scaled_identity', 'identity', 'zero' or 'none' has been violated.
		  
		  * -5. The constraints appear to be inconsistent.
		  
		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status
		  
		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

	*
		- m

		- is a scalar variable of type int, that holds the number of rows of :math:`A`.

	*
		- n

		- is a scalar variable of type int, that holds the number of columns of :math:`A`.

	*
		- A_ne

		- is a scalar variable of type int, that holds the number of nonzero entries in :math:`A`.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type int, that holds the column indices of :math:`A` in a row-wise storage scheme. The nonzeros must be ordered so that those in row i appear directly before those in row i+1, the order within each row is unimportant.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type int, that holds the starting position of each row of :math:`A`, as well as the total number of entries.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type double, that holds the values of the entries of the :math:`A` ordered as in A_col and A_ptr.

	*
		- b

		- is a one-dimensional array of size m and type double, that holds the linear term :math:`b` in the constraints. The i-th component of b, i = 0, ... , m-1, contains :math:`b_i`.

	*
		- n_depen

		- is a scalar variable of type int, that holds the number of dependent constraints, if any.

	*
		- depen

		- is a one-dimensional array of size m and type int, whose first n_depen components contain the indices of dependent constraints.

.. index:: pair: function; fdc_terminate
.. _doxid-galahad__fdc_8h_1a9c0167379258891dee32b35e0529b9f9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void fdc_terminate(
		void** data,
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>`* control,
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`fdc_inform_type <doxid-structfdc__inform__type>`)

