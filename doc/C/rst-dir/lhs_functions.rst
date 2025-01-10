.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_lhs_control_type.rst
	struct_lhs_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`lhs_control_type<doxid-structlhs__control__type>`;
	struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`;

	// global functions

	void :ref:`lhs_initialize<doxid-galahad__lhs_8h_1ae5e561917c238f90b8f6549a80c9d3d8>`(
		void **data,
		struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform
	);

	void :ref:`lhs_read_specfile<doxid-galahad__lhs_8h_1a38254f580fde3732f4f4e83e08180e63>`(struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control, const char specfile[]);

	void :ref:`lhs_ihs<doxid-galahad__lhs_8h_1a2a2e504e820685237f3ec3f8c97722ad>`(
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_dimen,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_points,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *seed,
		int* X,
		const struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform,
		void **data
	);

	void :ref:`lhs_get_seed<doxid-galahad__lhs_8h_1add3dc91a7fe9b311898e516798d81e14>`(int *seed);
	void :ref:`lhs_information<doxid-galahad__lhs_8h_1a5366dfb6b11cd47fbdb407ecbfcf60a9>`(void **data, struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`lhs_terminate<doxid-galahad__lhs_8h_1a24f8433561128e5c05e588d053b22f29>`(
		void **data,
		struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform
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

.. index:: pair: function; lhs_initialize
.. _doxid-galahad__lhs_8h_1ae5e561917c238f90b8f6549a80c9d3d8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_initialize(
		void **data,
		struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform
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

		- is a struct containing control information (see fit_control_type)

	*
		- inform

		- is a struct containing output information (see fit_inform_type)

.. index:: pair: function; lhs_read_specfile
.. _doxid-galahad__lhs_8h_1a38254f580fde3732f4f4e83e08180e63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_read_specfile(struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/lhs/LHS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/lhs.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- a struct containing control information (see above)

	*
		- specfile

		- a character string containing the name of the specfile

.. index:: pair: function; lhs_ihs
.. _doxid-galahad__lhs_8h_1a2a2e504e820685237f3ec3f8c97722ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_ihs(
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_dimen,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n_points,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *seed,
		int* X,
		const struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform,
		void **data
	)

The improved distributed hyper-cube sampling algorithm.

Discussion:

n_points points in an n_dimen dimensional Latin hyper-cube are to be
selected. Each of the coordinate dimensions is discretized to the values
1 through n. The points are to be chosen in such a way that no two
points have any coordinate value in common. This is a standard Latin
hypercube requirement, and there are many solutions.

This algorithm differs in that it tries to pick a solution which has the
property that the points are "spread out" as evenly as possible. It does
this by determining an optimal even spacing, and using the DUPLICATION
factor to allow it to choose the best of the various options available
to it.

Reference:

Brian Beachkofski, Ramana Grandhi, Improved Distributed Hypercube
Sampling, American Institute of Aeronautics and Astronautics Paper
2002-1274



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- n_dimen

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>` that specifies the spatial dimension

	*
		- n_points

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>` that specifies the number of points to be generated

	*
		- seed

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives a seed for the random number generator used

	*
		- X

		- is an array variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>` with dimensions [n_points][n_dimen] that gives the hyper-cube points

	*
		- control

		- 

	*
		- inform

		- 

	*
		- data

		- - see lhs_initialize

.. index:: pair: function; lhs_get_seed
.. _doxid-galahad__lhs_8h_1add3dc91a7fe9b311898e516798d81e14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_get_seed(int *seed)

Get a seed for the random number generator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- seed

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>` that gives the pseudorandom seed value.

.. index:: pair: function; lhs_information
.. _doxid-galahad__lhs_8h_1a5366dfb6b11cd47fbdb407ecbfcf60a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_information(void **data, struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lhs_terminate
.. _doxid-galahad__lhs_8h_1a24f8433561128e5c05e588d053b22f29:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void lhs_terminate(
		void **data,
		struct :ref:`lhs_control_type<doxid-structlhs__control__type>`* control,
		struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`lhs_control_type <doxid-structlhs__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)

