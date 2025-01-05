
callable functions
------------------

.. index:: pair: function; glrt_initialize
.. _doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; glrt_read_specfile
.. _doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/glrt/GLRT.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/glrt.pdf for a list of how these
keywords relate to the components of the control structure.


.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; glrt_import_control
.. _doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_import_control(T, INT, control, data, status)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

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

.. index:: pair: function; glrt_solve_problem
.. _doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_solve_problem(T, INT, data, status, n, power, weight, x, r, vector)

Solve the regularized-quadratic problem using reverse communication.



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

		  * **6**
                    the iteration is to be restarted with a larger
                    weight but with all other data unchanged. Set 
                    the argument ``r`` to $c$ for this entry.

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

		  * **4**
                    The iteration must be restarted. Reset the argument 
                    ``r`` to $c$ and re-enter with all other data unchanged.

		  * **-1**
                    an array allocation has failed

		  * **-2**
                    an array deallocation has failed

		  * **-3**
                    n and/or radius is not positive

		  * **-7**
                    the problem is unbounded from below. This can only
                    happen if power = 2, and in this case the objective
                    is unbounded along the arc x + t vector as t goes to
                    infinity

		  * **-15**
                    the matrix $M$ appears to be indefinite

		  * **-18**
                    the iteration limit has been exceeded

	*
		- n

		- is a scalar variable of type INT that holds the number of variables

	*
		- power

		- is a scalar of type T that holds the egularization power, $p \geq 2$

	*
		- weight

		- is a scalar of type T that holds the positive regularization weight, $\sigma$

	*
		- x

		- is a one-dimensional array of size n and type T that holds the solution $x$. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size n and type T that that must be set to $c$ on entry (status = 1) and re-entry (status = 4, 5). On exit, r contains the resiual $H x + c$.

	*
		- vector

		- is a one-dimensional array of size n and type T that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; glrt_information
.. _doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`glrt_inform_type <doxid-structglrt__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; glrt_terminate
.. _doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function glrt_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`glrt_inform_type <doxid-structglrt__inform__type>`)
