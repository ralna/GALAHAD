callable functions
------------------

.. index:: pair: function; nodend_initialize
.. _doxid-galahad__nodend_initialize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_initialize(INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; nodend_read_specfile
.. _doxid-galahad__nodend_read_specfile:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_read_specfile(INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/nodend/NODEND.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/nodend.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; nodend_order
.. _doxid-galahad__nodend_order:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_order(INT, control, data, status, n, perm, 
                              A_type, ne, A_row, A_col, A_ptr)

Order problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **1**
                    The order was successful, and the package is ready
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
                    One of the restrictions
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

		- is a scalar variable of type INT that holds the number of variables.

	*
		- perm

		- is a one-dimensional array of size n and type INT, that returns the computed permutation array, so that the perm[i]-th rows and columns in the permuted matrix $P A P^T$ correspond to those labelled i in $A$, 0 $\leq$ i $\leq$ n-1.

	*
		- A_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme<details-a_storage__sym>` used $A$. It should be one of 'coordinate' or 'sparse_by_rows'; lower or upper case variants are allowed. If A_type is not one of the supported values, the identity permutation will be returned.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of A in the sparse co-ordinate storage scheme. It need not be set for the other scheme.

	*
		- A_row

		- is a one-dimensional array of size ne and type INT that holds the row indices of the lower triangular part of $A$ in the sparse co-ordinate storage scheme. It need not be set for the other scheme, and in this case can be C_NULL

	*
		- A_col

		- is a one-dimensional array of size ne and type INT that holds the column indices of the lower triangular part of $A$.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of the lower triangular part of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other scheme is used, and in this case can be C_NULL

.. index:: pair: function; nodend_information
.. _doxid-galahad__nodend_information:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_information(INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`nodend_inform_type <doxid-structnodend__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

