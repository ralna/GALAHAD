.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_gltr_control_type.rst
	struct_gltr_inform_type.rst
	struct_gltr_time_type.rst

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
	struct_gltr_control_type.rst
	struct_gltr_inform_type.rst

Overview
~~~~~~~~



.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	-- modules

	:ref:`conf<doxid-namespaceconf>`

	-- table types

	:ref:`gltr_control_type<doxid-structgltr__control__type>`
	:ref:`gltr_inform_type<doxid-structgltr__inform__type>`

	-- functions

	function :ref:`gltr_initialize<doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c>`(data, control, status)
	function :ref:`gltr_read_specfile<doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31>`(control, specfile)
	function :ref:`gltr_import_control<doxid-galahad__gltr_8h_1acb8a654fc381e3f231c3d10858f111b3>`(control, data, status)
	function :ref:`gltr_solve_problem<doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18>`(data, status, n, radius, x, r, vector)
	function :ref:`gltr_information<doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb>`(data, inform, status)
	function :ref:`gltr_terminate<doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631>`(data, control, inform)

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Functions
---------

.. index:: pair: function; gltr_initialize
.. _doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_initialize(data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; gltr_read_specfile
.. _doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNGLTR.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/gltr.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; gltr_import_control
.. _doxid-galahad__gltr_8h_1acb8a654fc381e3f231c3d10858f111b3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_import_control(control, data, status)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; gltr_solve_problem
.. _doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_solve_problem(data, status, n, radius, x, r, vector)

Solve the trust-region problem using reverse communication.



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
		  
		  * 4. the iteration is to be restarted with a smaller radius but with all other data unchanged. Set r (below) to :math:`c` for this entry.
		  
		  Possible exit values are:
		  
		  * 0. the solution has been found
		  
		  * 2. the inverse of :math:`M` must be applied to vector with the result returned in vector and the function re-entered with all other data unchanged. This will only happen if control.unitm is false
		  
		  * 3. the product :math:`H` * vector must be formed, with the result returned in vector and the function re-entered with all other data unchanged
		  
		  * 5. The iteration must be restarted. Reset r (below) to :math:`c` and re-enter with all other data unchanged. This exit will only occur if control.steihaug_toint is false and the solution lies on the trust-region boundary
		  
		  * -1. an array allocation has failed
		  
		  * -2. an array deallocation has failed
		  
		  * -3. n and/or radius is not positive
		  
		  * -15. the matrix :math:`M` appears to be indefinite
		  
		  * -18. the iteration limit has been exceeded
		  
		  * -30. the trust-region has been encountered in Steihaug-Toint mode
		  
		  * -31. the function value is smaller than control.f_min

	*
		- n

		- is a scalar variable of type int, that holds the number of variables

	*
		- radius

		- is a scalar of type double, that holds the trust-region radius, :math:`\Delta`, used. radius must be strictly positive

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- r

		- is a one-dimensional array of size n and type double, that that must be set to :math:`c` on entry (status = 1) and re-entry ! (status = 4, 5). On exit, r contains the resiual :math:`H x + c`.

	*
		- vector

		- is a one-dimensional array of size n and type double, that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; gltr_information
.. _doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_information(data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`gltr_inform_type <doxid-structgltr__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; gltr_terminate
.. _doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function gltr_terminate(data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`gltr_inform_type <doxid-structgltr__inform__type>`)

