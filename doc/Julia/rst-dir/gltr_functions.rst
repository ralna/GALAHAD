callable functions
------------------

.. index:: pair: function; gltr_initialize
.. _doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; gltr_read_specfile
.. _doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/gltr/GLTR.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/gltr.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; gltr_import_control
.. _doxid-galahad__gltr_8h_1acb8a654fc381e3f231c3d10858f111b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_import_control(T, INT, control, data, status)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; gltr_solve_problem
.. _doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_solve_problem(T, INT, data, status, n, radius, x, r, vector)

Solve the trust-region problem using reverse communication.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the
		  entry and exit status from the package.

		  This **must** be set to

		  * **1**
                    on initial entry. Set the argument ``r`` (below) to $c$
                    for this entry.

		  * **4**
                    the iteration is to be restarted with a smaller
                    radius but with all other data unchanged. Set ``r``
                    to $c$ for this entry.

		  Possible exit values are:

		  * **0**
                    the solution has been found

		  * **2**
                    the inverse of $M$ must be applied to the argument 
                    ``vector`` (below) with the result returned in ``vector``
                    and the function re-entered with all other data unchanged. 
                    This will only happen if control.unitm is false

		  * **3**
                    the product of $H$ with ``vector`` must be formed, with the
                    result returned in ``vector`` and the function
                    re-entered with all other data unchanged

		  * **5**
                    The iteration must be restarted. Reset ``r`` to
                    $c$ and re-enter with all other data unchanged. This
                    exit will only occur if control.steihaug_toint is
                    false and the solution lies on the trust-region
                    boundary

		  * **-1**
                    an array allocation has failed

		  * **-2**
                    an array deallocation has failed

		  * **-3**
                    n and/or radius is not positive

		  * **-15**
                    the matrix $M$ appears to be indefinite

		  * **-18**
                    the iteration limit has been exceeded

		  * **-30**
                    the trust-region has been encountered in
                    Steihaug-Toint mode

		  * **-31**
                    the function value is smaller than control.f_min

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- radius

		- is a scalar of type T that holds the trust-region radius, $\Delta$, used. radius must be strictly positive

	*
		- x

		- is a one-dimensional array of size n and type T that holds the solution $x$. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size n and type T that that must be set to $c$ on entry (status = 1) and re-entry (status = 4, 5). On exit, r contains the resiual $H x + c$.

	*
		- vector

		- is a one-dimensional array of size n and type T that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; gltr_information
.. _doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`gltr_inform_type <doxid-structgltr__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; gltr_terminate
.. _doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gltr_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`gltr_inform_type <doxid-structgltr__inform__type>`)
