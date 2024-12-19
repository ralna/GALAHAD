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

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`glrt_control_type<doxid-structglrt__control__type>`;
	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`;

	// global functions

	void :ref:`glrt_initialize<doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32>`(
		void **data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`glrt_read_specfile<doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725>`(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		const char specfile[]
	);

	void :ref:`glrt_import_control<doxid-galahad__glrt_8h_1a722a069ab53a2f47dae17d01d6b505a1>`(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`glrt_solve_problem<doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` vector[]
	);

	void :ref:`glrt_information<doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d>`(void **data, struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`glrt_terminate<doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812>`(
		void **data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform
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

.. index:: pair: function; glrt_initialize
.. _doxid-galahad__glrt_8h_1a3a086b68a942ba1049f1d6a1b4724d32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_initialize(
		void **data,
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
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

		- is a struct containing control information (see :ref:`glrt_control_type <doxid-structglrt__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; glrt_read_specfile
.. _doxid-galahad__glrt_8h_1a4a436dfac6a63cf991cd629b3ed0e725:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_read_specfile(
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/glrt/GLRT.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/glrt.pdf for a list of how these keywords 
relate to the components of the control structure.

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
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; glrt_solve_problem
.. _doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_solve_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` power,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` weight,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` vector[]
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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  This must be set to
		  
		  * **1**
                    on initial entry. Set r (below) to $c$ for
                    this entry.
		  
		  * **6**
                    the iteration is to be restarted with a larger
                    weight but with all other data unchanged. Set r
                    (below) to $c$ for this entry.
		  
		  Possible exit values are:
		  
		  * **0**
                    the solution has been found
		  
		  * **2**
                    the inverse of $M$ must be applied to vector
                    with the result returned in vector and the function
                    re-entered with all other data unchanged. This will
                    only happen if control.unitm is false
		  
		  * **3**
                    the product $H$ \* **vector must be formed, with
                    the result returned in vector and the function
                    re-entered with all other data unchanged
		  
		  * 4**
                    The iteration must be restarted. Reset r (below) to
                    $c$ and re-enter with all other data
                    unchanged.
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- power

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the egularization power, $p \geq 2$

	*
		- weight

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the positive regularization weight, $\sigma$

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the solution $x$. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that that must be set to $c$ on entry (status = 1) and re-entry (status = 4, 5). On exit, r contains the resiual $H x + c$.

	*
		- vector

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; glrt_information
.. _doxid-galahad__glrt_8h_1a3570dffe8910d5f3cb86020a65566c8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_information(void **data, struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; glrt_terminate
.. _doxid-galahad__glrt_8h_1a107fe137aba04a93fdbcbb0b9e768812:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void glrt_terminate(
		void **data,
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

