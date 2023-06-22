.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_glrt_control_type.rst
	struct_glrt_inform_type.rst
	struct_glrt_time_type.rst

.. _details-global:

function calls
--------------

.. ---------------------------------------------------------------------------
.. _global:

Global Scope
============


.. toctree::
	:hidden:

	namespace_conf.rst
	struct_glrt_control_type.rst
	struct_glrt_inform_type.rst

Overview
~~~~~~~~



.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	-- modules

	:ref:`conf<doxid-namespaceconf>`

	-- table types

	:ref:`glrt_control_type<doxid-structglrt__control__type>`
	:ref:`glrt_inform_type<doxid-structglrt__inform__type>`

	-- functions

	function :ref:`glrt_initialize<doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32>`(data, control, status)
	function :ref:`glrt_read_specfile<doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725>`(control, specfile)
	function :ref:`glrt_import_control<doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1>`(control, data, status)
	function :ref:`glrt_solve_problem<doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>`(data, status, n, power, weight, x, r, vector)
	function :ref:`glrt_information<doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d>`(data, inform, status)
	function :ref:`glrt_terminate<doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812>`(data, control, inform)

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Functions
---------

.. index:: pair: function; glrt_initialize
.. _doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_initialize(data, control, status)

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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_read_specfile(control, specfile)

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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_import_control(control, data, status)

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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_solve_problem(data, status, n, power, weight, x, r, vector)

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
		  
		  * 3. the product :math:`H` * vector must be formed, with the result returned in vector and the function re-entered with all other data unchanged
		  
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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_information(data, inform, status)

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

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function glrt_terminate(data, control, inform)

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

