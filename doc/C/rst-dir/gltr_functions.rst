.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_gltr_control_type.rst
	struct_gltr_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>`;
	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>`;

	// global functions

	void :ref:`gltr_initialize<doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c>`(
		void **data,
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`gltr_read_specfile<doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31>`(
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		const char specfile[]
	);

	void :ref:`gltr_import_control<doxid-galahad__gltr_8h_1acb8a654fc381e3f231c3d10858f111b3>`(
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`gltr_solve_problem<doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` vector[]
	);

	void :ref:`gltr_information<doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb>`(void **data, struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`gltr_terminate<doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631>`(
		void **data,
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>`* inform
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

.. index:: pair: function; gltr_initialize
.. _doxid-galahad__gltr_8h_1ac06a7060d9355146e801157c2f29ca5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_initialize(
		void **data,
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
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

		- is a struct containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; gltr_read_specfile
.. _doxid-galahad__gltr_8h_1a68a3273a88b27601e72b61f10a23de31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_read_specfile(
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		const char specfile[]
	)

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/gltr/GLTR.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/gltr.pdf for a list of how these keywords 
relate to the components of the control structure.

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

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_import_control(
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * 1. The import was successful, and the package is ready for the solve phase

.. index:: pair: function; gltr_solve_problem
.. _doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_solve_problem(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>` radius,
		:ref:`rpc_<doxid-galahad__rpc_8h_>` x[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` r[],
		:ref:`rpc_<doxid-galahad__rpc_8h_>` vector[]
	)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  This must be set to
		  
		  * **1**
                    on initial entry. Set r (below) to $c$ for
                    this entry.
		  
		  * **4**
                    the iteration is to be restarted with a smaller
                    radius but with all other data unchanged. Set r
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
		  
		  * 5**
                    The iteration must be restarted. Reset r (below) to
                    $c$ and re-enter with all other data
                    unchanged. This exit will only occur if
                    control.steihaug_toint is false and the solution
                    lies on the trust-region boundary
		  
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

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables

	*
		- radius

		- is a scalar of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the trust-region radius, $\Delta$, used. radius must be strictly positive

	*
		- x

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the solution $x$. The j-th component of x, j = 0, ... , n-1, contains $x_j$.

	*
		- r

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that that must be set to $c$ on entry (status = 1) and re-entry ! (status = 4, 5). On exit, r contains the resiual $H x + c$.

	*
		- vector

		- is a one-dimensional array of size n and type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that should be used and reset appropriately when status = 2 and 3 as directed.

.. index:: pair: function; gltr_information
.. _doxid-galahad__gltr_8h_1a1b1b4d87884833c4bfe184ff79c1e2bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_information(void **data, struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; gltr_terminate
.. _doxid-galahad__gltr_8h_1ac3e0cbd0ecc79b37251fad7fd6f47631:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void gltr_terminate(
		void **data,
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>`* control,
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>`* inform
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

		- is a struct containing control information (see :ref:`gltr_control_type <doxid-structgltr__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`gltr_inform_type <doxid-structgltr__inform__type>`)

