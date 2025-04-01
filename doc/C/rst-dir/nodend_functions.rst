.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_nodend_control_type.rst
	struct_nodend_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`conf<doxid-namespaceconf>`;

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`nodend_control_type<doxid-structnodend__control__type>`;
	struct :ref:`nodend_inform_type<doxid-structnodend__inform__type>`;

	// global functions

	void :ref:`nodend_initialize<doxid-galahad__nodend_8h_initialize>`(
		void **data,
		struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`nodend_read_specfile<doxid-galahad__nodend_8h_read_specfile>`(struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control, const char specfile[]);

	void :ref:`nodend_order<doxid-galahad__nodend_8h_order>`(
		struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` perm[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[]
	);

	void :ref:`nodend_information<doxid-galahad__nodend_8h_information>`(
		void **data,
		struct :ref:`nodend_inform_type<doxid-structnodend__inform__type>`* inform,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; ipc_
.. _doxid-galahad__ipc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int ipc_

``ipc_`` is the default integer word length used, but may be changed to 
``int64_t`` by defining the  preprocessor variable ``INTEGER_64``.

function calls
--------------

.. index:: pair: function; nodend_initialize
.. _doxid-galahad__nodend_8h_initialize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nodend_initialize(
		void **data,
		struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control,
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

		- is a struct containing control information (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; nodend_read_specfile
.. _doxid-galahad__nodend_8h_read_specfile:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nodend_read_specfile(struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values 
associated with given keywords to the corresponding control 
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords with
associated default values is provided in \$GALAHAD/src/nodend/NODEND.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/nodend.pdf for a list of how these keywords relate 
to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; nodend_order
.. _doxid-galahad__nodend_8h_order:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nodend_order(
		struct :ref:`nodend_control_type<doxid-structnodend__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` perm[],
		const char H_type[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_col[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` A_ptr[]
	)

Find a row/colum permutation for sparse Cholesky-like factorization.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase
		  
		  * **-1**
                    An allocation error occurred. A message indicating
                    the offending array is written on unit
                    control.error, and the returned allocation status
                    and a string containing the name of the offending
                    array are held in inform.alloc_status and
                    inform.bad_alloc respectively.
		  
		  * **-2**
                    A deallocation error occurred. A message indicating
                    the offending array is written on unit control.error
                    and the returned allocation status and a string
                    containing the name of the offending array are held
                    in inform.alloc_status and inform.bad_alloc
                    respectively.
		  
		  * **-3**
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'coordinate' or
                    'sparse_by_rows' has been violated. One of the restrictions
                    n $> 0$, A.n $> 0$ or A.ne $< 0$, for co-ordinate entry,
                    or requirements that A.type contain its relevant string
                    'coordinate' or 'sparse_by_rows, and
                    control.version in one of '4.0', '5.1' or '5.2'
                    has been violated.
		  
		  * **-26**
                    The requested version of METIS is not available.
		  
		  * **-57**
                    METIS has insufficient memory to continue.
		  
		  * **-71**
                    An internal METIS error occurred.

	*
		- n

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of variables.

	*
		- perm

		- is a one-dimensional array of size n and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that returns the computed permutation array, so that the perm[i]-th rows and columns in the permuted matrix $P A P^T$ correspond to those labelled i in $A$, 0 $\leq$ i $\leq$ n-1.

	*
		- A_type

		- is a one-dimensional array of type char that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate' or 'sparse_by_rows'; lower or upper case variants are allowed. If A_type is not one of the supported values, the identity permutation will be returned.

	*
		- ne

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- A_row

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the row indices of the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for the other scheme, and in this case can be NULL

	*
		- A_col

		- is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the column indices of the lower triangular part of $A$.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the starting position of each row of the lower triangular part of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other scheme is used, and in this case can be NULL

.. index:: pair: function; nodend_information
.. _doxid-galahad__nodend_8h_information:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void nodend_information(
		void **data,
		struct :ref:`nodend_inform_type<doxid-structnodend__inform__type>`* inform,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`nodend_inform_type <doxid-structnodend__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully


