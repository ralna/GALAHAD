callable functions
------------------

.. index:: pair: function; bsc_initialize
.. _doxid-galahad__bsc_initialize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; bsc_read_specfile
.. _doxid-galahad__bsc_read_specfile:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.
An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/bsc/BSC.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/bsc.pdf for a list of how these keywords relate to the 
components of the control structure.

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

        *
                - specfile

                - is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; bsc_import
.. _doxid-galahad__bsc_import:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function bsc_import(T, INT, control, data, status, m, n, 
                            A_type, A_ne, A_row, A_col, A_ptr, S_ne)

Import problem data into internal storage prior to solution.


.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure whose members provide control parameters for the remaining procedures (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

        *
                - data

                - holds private internal data

        *
                - status

                - is a scalar variable of type INT that gives the exit
                  status from the package. Possible values are:

                  * **0**
                    The import was successful

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
                    The restrictions n > 0 or m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate' or 'sparse_by_rows' has been violated.

        *
                - m

                - is a scalar variable of type INT that holds the number of rows of $A$

        *
                - n

                - is a scalar variable of type INT that holds the number of columns of $A$

        *
                - A_type

                - is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme<details-asc_storage__unsym>` used for the matrix $A$. It should be one of 'coordinate', 'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

        *
                - A_ne

                - is a scalar variable of type INT that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

        *
                - A_row

                - is a one-dimensional array of size A_ne and type INT that holds the row indices of $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes, and in this case can be C_NULL.

        *
                - A_col

                - is a one-dimensional array of size A_ne and type INT that holds the column indices of $A$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be C_NULL.

        *
                - A_ptr

                - is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

        *
                - S_ne

                - is a scalar variable of type INT that gives the number of entries in $S$.

.. index:: pair: function; bsc_reset_control
.. _doxid-galahad__bsc_reset_control:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function bsc_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure whose members provide control parameters for the remaining procedures (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

        *
                - data

                - holds private internal data

        *
                - status

                - is a scalar variable of type INT that gives the exit
                  status from the package. Possible values are:

                  * **0**
                    The import was successful.

.. index:: pair: function; bsc_form_s
.. _doxid-galahad__bsc_form_s:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function bsc_form_s(T, INT, data, status, m, n, a_ne, A_val, 
                            S_ne, S_row, S_col, S_ptr, S_val, D)

Form the Schur complement matrix, $S$.

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
                    The restrictions n > 0 or m > 0 or requirement that
                    a type contains its relevant string 'dense',
                    'coordinate' or 'sparse_by_rows' has been violated.

        *
                - m

                - is a scalar variable of type INT that holds the number of rows of $A$

        *
                - n

                - is a scalar variable of type INT that holds the number of columns of $A$


        *
                - A_ne

                - is a scalar variable of type INT that holds the number of entries in $A$.

        *
                - A_val

                - is a one-dimensional array of size a_ne and type T that holds the values of the entries of the matrix $A$ in any of the available storage schemes.

        *
                - S_ne

                - is a scalar variable of type INT that holds the number of entries in the lower traingle of $S$ in the sparse co-ordinate storage scheme.

        *
                - S_row

                - is a one-dimensional array of size S_ne and type INT that gives the row indices the lower traingle of $S$ in the sparse co-ordinate storage scheme.

        *
                - S_col

                - is a one-dimensional array of size S_ne and type INT that gives the column indices the lower traingle of $S$ in either the sparse co-ordinate, or the sparse row-wise storage scheme. 

        *
                - S_ptr

                - is a one-dimensional array of size n+1 and type INT that gives the starting position of each row the lower traingle of $S$, as well as the total number of entries, in the sparse row-wise storage scheme. If this scheme is not wanted, S_ptr can be NULL to save storage.

        *
                - S_val

                - is a one-dimensional array of size S_ne and type T that gives the values of the entries of the lower traingle of the matrix $S$.

        *
                - D

                - is a one-dimensional array of size n and type T that gives the values of the diagonal entries in $D$. If $D$ is the identity matrix, D can be NULL to save storage.


.. index:: pair: function; bsc_information
.. _doxid-galahad__bsc_information:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; bsc_terminate
.. _doxid-galahad__bsc_terminate:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)
