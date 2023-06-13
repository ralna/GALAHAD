.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_lstr_control_type.rst
	struct_lstr_inform_type.rst
	struct_lstr_time_type.rst

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
	struct_lstr_control_type.rst
	struct_lstr_inform_type.rst

Overview
~~~~~~~~



.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	-- modules

	:ref:`conf<doxid-namespaceconf>`

	-- table types

	:ref:`lstr_control_type<doxid-structlstr__control__type>`
	:ref:`lstr_inform_type<doxid-structlstr__inform__type>`

	-- functions

	function :ref:`lstr_initialize<doxid-galahad__lstr_8h_1ae423bf7ffc77c89f461448ca1f5c286c>`(data, control, status)
	function :ref:`lstr_read_specfile<doxid-galahad__lstr_8h_1a3d3fa989fe4c3b40cd7e296249d2205d>`(control, specfile)
	function :ref:`lstr_import_control<doxid-galahad__lstr_8h_1a1a8ad63d944dc046fd2040554d6d01e5>`(control, data, status)
	function :ref:`lstr_solve_problem<doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562>`(data, status, m, n, radius, x, u, v)
	function :ref:`lstr_information<doxid-galahad__lstr_8h_1a5929f00ea00af253ede33a6749451481>`(data, inform, status)
	function :ref:`lstr_terminate<doxid-galahad__lstr_8h_1aa198189942e179e52699e1fedfcdf9d1>`(data, control, inform)

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Functions
---------

.. index:: pair: function; lstr_initialize
.. _doxid-galahad__lstr_8h_1ae423bf7ffc77c89f461448ca1f5c286c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_initialize(data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The import was succesful.

.. index:: pair: function; lstr_read_specfile
.. _doxid-galahad__lstr_8h_1a3d3fa989fe4c3b40cd7e296249d2205d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNLSTR.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/lstr.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; lstr_import_control
.. _doxid-galahad__lstr_8h_1a1a8ad63d944dc046fd2040554d6d01e5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_import_control(control, data, status)

Import control parameters prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; lstr_solve_problem
.. _doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_solve_problem(data, status, m, n, radius, x, u, v)

Solve the trust-region least-squares problem using reverse communication.



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
		  
		  * 5. the iteration is to be restarted with a smaller radius but with all other data unchanged. Set u (below) to :math:`b` for this entry.
		  
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
		  
		  * -3. one or more of n, m or weight violates allowed bounds
		  
		  * -18. the iteration limit has been exceeded
		  
		  * -25. status is negative on entry

	*
		- m

		- is a scalar variable of type int, that holds the number of equations (i.e., rows of :math:`A`), :math:`m > 0`

	*
		- n

		- is a scalar variable of type int, that holds the number of variables (i.e., columns of :math:`A`), :math:`n > 0`

	*
		- radius

		- is a scalar of type double, that holds the trust-region radius, :math:`\Delta > 0`

	*
		- x

		- is a one-dimensional array of size n and type double, that holds the solution :math:`x`. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- u

		- is a one-dimensional array of size m and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type double, that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; lstr_information
.. _doxid-galahad__lstr_8h_1a5929f00ea00af253ede33a6749451481:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_information(data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`lstr_inform_type <doxid-structlstr__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type int, that gives the exit status from the package. Possible values are (currently):
		  
		  * 0. The values were recorded succesfully

.. index:: pair: function; lstr_terminate
.. _doxid-galahad__lstr_8h_1aa198189942e179e52699e1fedfcdf9d1:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	function lstr_terminate(data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`lstr_inform_type <doxid-structlstr__inform__type>`)

