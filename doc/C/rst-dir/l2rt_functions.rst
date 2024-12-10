.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_l2rt_control_type.rst
	struct_l2rt_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`;
	struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`;

	// global functions

	void :ref:`l2rt_initialize<doxid-galahad__l2rt_8h_1a0103448a3db662f9c483f9f44a5112bc>`(
		void **data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`l2rt_read_specfile<doxid-galahad__l2rt_8h_1a1b63f8b501208629cceb662b03f35684>`(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		const char specfile[]
	);

	void :ref:`l2rt_import_control<doxid-galahad__l2rt_8h_1adf880b26c8aea32493857f8576e58ae8>`(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`l2rt_solve_problem<doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` shift,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[]
	);

	void :ref:`l2rt_information<doxid-galahad__l2rt_8h_1a4fa18245556cf87b255b2b9ac5748ca9>`(void **data, struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`l2rt_terminate<doxid-galahad__l2rt_8h_1aa9b62de33c3d6c129cca1e90a3d548b7>`(
		void **data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform
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

.. index:: pair: function; l2rt_initialize
.. _doxid-galahad__l2rt_8h_1a0103448a3db662f9c483f9f44a5112bc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_initialize(
		void **data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
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

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; l2rt_read_specfile
.. _doxid-galahad__l2rt_8h_1a1b63f8b501208629cceb662b03f35684:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_read_specfile(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/l2rt/L2RT.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/l2rt.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; l2rt_import_control
.. _doxid-galahad__l2rt_8h_1adf880b26c8aea32493857f8576e58ae8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_import_control(
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Import control parameters prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; l2rt_solve_problem
.. _doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_solve_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` shift,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` u[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` v[]
	)

Solve the regularized-least-squares problem using reverse communication.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  This must be set to
		  
		  * **1**
                    on initial entry. Set u (below) to $b$ for
                    this entry.
		  
		  Possible exit values are:
		  
		  * **0**
                    the solution has been found
		  
		  * **2**
                    The user must perform the operation
		    
		    .. math::
		    
		    	u := u + A v,
		    
		    \n
		                   u := u + A v,
		    \n and recall the function. The vectors $u$ and $v$ are available in the arrays u and v (below) respectively, and the result $u$ must overwrite the content of u. No argument except u should be altered before recalling the function
		  
		  * **3**
                    The user must perform the operation
		    
		    .. math::
		    
		    	v := v + A^T u,
		    
		    \n
		                   v := v + A^T u,
		    \n and recall the function. The vectors $u$ and $v$ are available in the arrays u and v (below) respectively, and the result $v$ must overwrite the content of v. No argument except v should be altered before recalling the function
		  
		  * **4**
                    The user must reset u (below) to $b$ are
                    recall the function. No argument except u should be
                    altered before recalling the function
		  
		  * **-1**
                    an array allocation has failed
		  
		  * **-2**
                    an array deallocation has failed
		  
		  * **-3**
                    one or more of n, m, weight or shift violates
                    allowed bounds
		  
		  * **-18**
                    the iteration limit has been exceeded
		  
		  * **-25**
                    status is negative on entry

	*
		- m

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of equations (i.e., rows of $A$), $m > 0$

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables (i.e., columns of $A$), $n > 0$

	*
		- power

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the regularization power, $p \geq 2$

	*
		- weight

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the regularization weight, $\sigma > 0$

	*
		- shift

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the shift, $\mu$

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the solution $x$. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- u

		- is a one-dimensional array of size m and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; l2rt_information
.. _doxid-galahad__l2rt_8h_1a4fa18245556cf87b255b2b9ac5748ca9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_information(void **data, struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`l2rt_inform_type <doxid-structl2rt__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; l2rt_terminate
.. _doxid-galahad__l2rt_8h_1aa9b62de33c3d6c129cca1e90a3d548b7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void l2rt_terminate(
		void **data,
		struct :ref:`l2rt_control_type<doxid-structl2rt__control__type>`* control,
		struct :ref:`l2rt_inform_type<doxid-structl2rt__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`l2rt_control_type <doxid-structl2rt__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`l2rt_inform_type <doxid-structl2rt__inform__type>`)

