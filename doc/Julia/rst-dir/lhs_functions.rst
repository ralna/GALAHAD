callable functions
------------------

.. index:: pair: function; lhs_initialize
.. _doxid-galahad__lhs_8h_1ae5e561917c238f90b8f6549a80c9d3d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_initialize(T, INT, data, control, inform)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lhs_control_type <doxid-structlhs__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)

.. index:: pair: function; lhs_read_specfile
.. _doxid-galahad__lhs_8h_1a38254f580fde3732f4f4e83e08180e63:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/lhs/LHS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/lhs.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lhs_control_type <doxid-structlhs__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; lhs_ihs
.. _doxid-galahad__lhs_8h_1a2a2e504e820685237f3ec3f8c97722ad:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_ihs(T, INT, n_dimen, n_points, seed, X, control, inform, data)

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

Brian Beachkofski, Ramana Grandhi, Improved Distributed Hypercube Sampling, American Institute of Aeronautics and Astronautics Paper 2002-1274



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- n_dimen

		- is a scalar variable of type int that specifies the spatial dimension

	*
		- n_points

		- is a scalar variable of type int that specifies the number of points to be generated

	*
		- seed

		- is a scalar variable of type INT that gives a seed for the random number generator used

	*
		- X

		- is an array variable of type int with dimensions [n_dimen][n_points] that gives the hyper-cube points


	*
		- control

		- is a structure containing control information (see :ref:`lhs_control_type <doxid-structlhs__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)


	*
		- data

		- - holds private internal data

.. index:: pair: function; lhs_get_seed
.. _doxid-galahad__lhs_8h_1add3dc91a7fe9b311898e516798d81e14:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_get_seed(T, INT, seed)

Get a seed for the random number generator.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- seed

		- is a scalar variable of type int that gives the pseudorandom seed value.

.. index:: pair: function; lhs_information
.. _doxid-galahad__lhs_8h_1a5366dfb6b11cd47fbdb407ecbfcf60a9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lhs_terminate
.. _doxid-galahad__lhs_8h_1a24f8433561128e5c05e588d053b22f29:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lhs_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lhs_control_type <doxid-structlhs__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lhs_inform_type <doxid-structlhs__inform__type>`)
