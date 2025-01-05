callable functions
------------------

.. index:: pair: function; fdc_initialize
.. _doxid-galahad__fdc_8h_1a09ed47873fc4b54eac5b10958939459b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fdc_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The import was successful.

.. index:: pair: function; fdc_read_specfile
.. _doxid-galahad__fdc_8h_1aa5e20e6a3ed015cdd927c1bfc7f00a2a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fdc_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/fdc/FDC.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/fdc.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; fdc_find_dependent_rows
.. _doxid-galahad__fdc_8h_1a37ea723b9a1b8799e7971344858d020a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fdc_find_dependent_rows(T, INT, control, data, inform, status, 
                                         m, n, A_ne, A_col, A_ptr, A_val, b, 
                                         n_depen, depen)

Find dependent rows and, if any, check if $A x = b$ is consistent

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`fdc_inform_type <doxid-structfdc__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the
		  entry and exit status from the package.

		  Possible exit values are:

		  * **0**
                    The run was successful.

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
                    The restrictions n > 0 and m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.

		  * **-5**
                    The constraints appear to be inconsistent.

		  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status

		  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.

	*
		- m

		- is a scalar variable of type INT that holds the number of rows of $A$.

	*
		- n

		- is a scalar variable of type INT that holds the number of columns of $A$.

	*
		- A_ne

		- is a scalar variable of type INT that holds the number of nonzero entries in $A$.

	*
		- A_col

		- is a one-dimensional array of size A_ne and type INT that holds the column indices of $A$ in a row-wise storage scheme. The nonzeros must be ordered so that those in row i appear directly before those in row i+1, the order within each row is unimportant.

	*
		- A_ptr

		- is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $A$, as well as the total number of entries.

	*
		- A_val

		- is a one-dimensional array of size a_ne and type T that holds the values of the entries of the $A$ ordered as in A_col and A_ptr.

	*
		- b

		- is a one-dimensional array of size m and type T that holds the linear term $b$ in the constraints. The i-th component of ``b``, i = 1, ... , m, contains $b_i$.

	*
		- n_depen

		- is a scalar variable of type INT that holds the number of dependent constraints, if any.

	*
		- depen

		- is a one-dimensional array of size m and type INT whose first n_depen components contain the indices of dependent constraints.

.. index:: pair: function; fdc_terminate
.. _doxid-galahad__fdc_8h_1a9c0167379258891dee32b35e0529b9f9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fdc_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`fdc_control_type <doxid-structfdc__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`fdc_inform_type <doxid-structfdc__inform__type>`)
